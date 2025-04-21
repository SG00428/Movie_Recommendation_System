import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import pickle
from models.ncf import NCF

ratings = pd.read_csv("data/Netflix_Dataset_Rating.csv")
ratings = ratings.sample(frac=0.1, random_state=42)  # Use a smaller sample for faster training
user_enc = LabelEncoder()
item_enc = LabelEncoder()
ratings['user_enc'] = user_enc.fit_transform(ratings['User_ID'])
ratings['item_enc'] = item_enc.fit_transform(ratings['Movie_ID'])
print("doing")
with open("utils/encoders.pkl", "wb") as f:
    pickle.dump((user_enc, item_enc), f)

X_users = torch.tensor(ratings['user_enc'].values, dtype=torch.long)
X_items = torch.tensor(ratings['item_enc'].values, dtype=torch.long)
Y = torch.tensor(ratings['Rating'].values, dtype=torch.float32)

train_data = TensorDataset(X_users, X_items, Y)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
print("doing")
model = NCF(len(user_enc.classes_), len(item_enc.classes_))
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("doing")
for epoch in range(3):
    total_loss = 0
    for user, item, label in train_loader:
        optimizer.zero_grad()
        output = model(user, item)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
print("doing")
torch.save(model.state_dict(), "models/ncf_model.pth")