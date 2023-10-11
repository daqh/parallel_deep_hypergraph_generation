import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from dgmh.utils import compute_embeddings, compute_edge_index
import matplotlib.pyplot as plt

def train_models(autoencoder: nn.Module, hsdm: nn.Module, hyperedges, epochs: int, device: str):
    nodes_one_hot, X, y = compute_embeddings(hyperedges)
    edge_index = compute_edge_index(hyperedges)

    X = X.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

    train_pos_nodes = torch.where(X_train > 0)
    test_pos_nodes = torch.where(X_test > 0)
    neg_nodes = torch.where(~(X > 0))

    autoencoder.train()
    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

    for epoch in range(epochs):
        autoencoder.train()
        autoencoder_optimizer.zero_grad()

        encoded = autoencoder.encode(nodes_one_hot, edge_index)
        decoded = autoencoder.decode(encoded)

        autoencoder.eval()
        s = 0
        for a, h in zip(decoded, hyperedges):
            cat = torch.distributions.Categorical(torch.softmax(a * 200, dim=0))
            node = cat.sample()
            if node + 1 in h:
                s += 1
        autoencoder.train()

        reconstruction_loss = autoencoder.recon_loss(decoded, train_pos_nodes, neg_nodes)
        kl_loss = 1 / X.size(0) * autoencoder.kl_loss()

        loss = reconstruction_loss + kl_loss
        loss.backward()

        autoencoder_optimizer.step()
        with torch.inference_mode():
            autoencoder.eval()
            
            encoded = autoencoder.encode(nodes_one_hot, edge_index)
            decoded = autoencoder.decode(encoded)

            # Sample negative nodes (as many as positive nodes)
            test_neg_index = torch.randint(0, len(neg_nodes[0]), (len(test_pos_nodes[0]),))
            epoch_neg_nodes = (neg_nodes[0][test_neg_index], neg_nodes[1][test_neg_index])

            y_pred = torch.cat((decoded[test_pos_nodes], decoded[epoch_neg_nodes]), dim=0)
            y_true = torch.cat((torch.ones(len(test_pos_nodes[0])), torch.zeros(len(epoch_neg_nodes[0]))), dim=0)

            ap = average_precision_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            roc_auc = roc_auc_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item()} - Sampled {s} nodes correctly - AP: {ap} - ROC-AUC: {roc_auc}")

    autoencoder.eval()
    hsdm.train()

    encoded = autoencoder.encode(nodes_one_hot, edge_index)
    decoded = autoencoder.decode(encoded)

    X_train, X_test, y_train, y_test = train_test_split(decoded, y, stratify=y, random_state=42, test_size=0.2)

    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_resample(X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())
    X_resampled = torch.from_numpy(X_resampled).to(device)
    y_resampled = torch.from_numpy(y_resampled).to(device).type(torch.float32)

    hsdm_optimizer = torch.optim.Adam(hsdm.parameters(), lr=0.0001, weight_decay=5e-4)
    hsdm_criterion = nn.BCELoss()
    for epoch in range(epochs * 2):
        hsdm_optimizer.zero_grad()
        pred = hsdm(X_resampled)
        loss = hsdm_criterion(pred, y_resampled)
        pred = pred.round()
        loss.backward()
        hsdm_optimizer.step()
        if epoch % 2 == 0:
            with torch.inference_mode():
                pred = hsdm(X_test)
                f1 = f1_score(y_test.argmax(dim=1).detach().cpu().numpy(), pred.argmax(dim=1).detach().cpu().numpy(), average='weighted')
                print(f'Epoch {epoch+1}/{epochs} {loss.item()} - F1 score: {f1}')
