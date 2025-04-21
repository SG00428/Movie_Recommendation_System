import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from models.ncf import NCF
from models.ensemble import FusionModel
import pickle

ratings = pd.read_csv("data/Netflix_Dataset_Rating.csv")
ratings = ratings.sample(frac=0.1, random_state=42)
user_enc, item_enc = pickle.load(open("utils/encoders.pkl", "rb"))
ratings['user_enc'] = user_enc.transform(ratings['User_ID'])
ratings['item_enc'] = item_enc.transform(ratings['Movie_ID'])

# Load models
svd_model = pickle.load(open("models/svd_model.pkl", "rb"))
ncf_model = NCF(len(user_enc.classes_), len(item_enc.classes_))
ncf_model.load_state_dict(torch.load("models/ncf_model.pth"))
ncf_model.eval()

mf_scores = []
ncf_scores = []
targets = []

for _, row in ratings.iterrows():
    uid, iid, rating = row['user_enc'], row['item_enc'], row['Rating']
    mf_scores.append(svd_model.predict(uid, iid).est)
    with torch.no_grad():
        ncf_score = ncf_model(torch.LongTensor([uid]), torch.LongTensor([iid])).item()
    ncf_scores.append(ncf_score)
    targets.append(rating)
print("doing")
X1 = torch.tensor(mf_scores,dtype=torch.float32)
X2 = torch.tensor(ncf_scores,dtype=torch.float32)
Y = torch.tensor(targets,dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X1, X2, Y), batch_size=64, shuffle=True)

fusion_model = FusionModel()
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for epoch in range(3):
    total_loss = 0
    for x1, x2, y in train_loader:
        output = fusion_model(x1, x2)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
print("doing")
torch.save(fusion_model.state_dict(), "models/fusion_model.pth")