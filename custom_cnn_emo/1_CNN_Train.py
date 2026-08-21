criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

epochs = 5
model.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
        
    print(f"Epoch {epoch+1}/{epochs} -> Loss: {epoch_loss/total:.4f} | Acc: {correct/total * 100:.2f}%")
