import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

# Defining the operations to do in a training step
def train_step(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer):
    # Put model in train mode
    model.train()
    # Setup train loss and train accuracy values
    train_loss,train_acc = 0,0
    # Loop through DataLoader batches
    for batch,sample_batched in enumerate(dataloader):
        # Send data to target device
        X, y = sample_batched
        X.requires_grad = True
        # Forward pass
        y_pred = model(X)
        # Calculate  and accumulate loss
        loss = loss_fn(y_pred,y)
        train_loss += loss.item()
        # Optimizer zero grad
        optimizer.zero_grad()
        # Loss backward
        loss.backward()
        # Optimizer step
        optimizer.step()
        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss,train_acc
from sklearn.metrics import f1_score
# Defining the operations to do in a validation step
def val_step(model:torch.nn.Module,
              dataloader:torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module):
    bcm = BinaryConfusionMatrix()
    # Put model in eval mode
    model.eval()
    # Setup validation loss and validation accuracy values
    val_loss,val_acc = 0,0
    y_pred = torch.IntTensor()
    y_true = torch.IntTensor()
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch,sample_batched in enumerate(dataloader):
            # Send data to target device
            X,y = sample_batched

            # Forward pass
            val_pred_logits = model(X)
            # Calculate and accumulate loss
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()
            # Calculate and accumulate accuracy
            val_pred_labels = val_pred_logits.argmax(dim=1)
            val_acc += ((val_pred_labels == y).sum().item()/len(val_pred_labels))
            y_pred=torch.cat((y_pred,val_pred_labels.cpu().detach()))
            y_true=torch.cat((y_true,y.cpu().detach()))
    # Adjust metrics to get average loss and accuracy per batch
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    f1 = f1_score(y_true, y_pred)
    # conf_mat = bcm(y_pred.flatten(), y_true.flatten())
    return val_loss,val_acc, f1, ""
    




# Taking in various parameters required for training and validation steps
def train(model:torch.nn.Module,
          train_dataloader:torch.utils.data.DataLoader,
          val_dataloader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          loss_fn:torch.nn.Module = nn.CrossEntropyLoss(),
          epochs:int = 5,
          split:int = 0,
          dir = ""):
    
    # Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    # Instantiating the best validation accuracy
    best_val = 0
    # Loop through training and validation steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        val_loss, val_acc, _,_ = val_step(model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn)
        # Saving the model obtaining the best validation accuracy through the epochs
        if val_acc > best_val:
            best_val = val_acc
            checkpoint = {"model": model,
                          "state_dict": model.state_dict(),
                          "optimizer": optimizer.state_dict()}
            checkpoint_name = f"{dir}/checkpoint_"+str(split)+".pth"
            torch.save(checkpoint, checkpoint_name)
        # else:
        #     continue
        # Print out what's happening
        # print(
        #     f"Epoch: {epoch+1} | "
        #     f"train_loss: {train_loss:.4f} | "
        #     f"train_acc: {train_acc:.4f} | "
        #     f"val_loss: {val_loss:.4f} | "
        #     f"val_acc: {val_acc:.4f}"
        # )
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
    # Return the filled results at the end of the epochs
    return results
    
#-----------ONE-------------------


# TRAINING

# Defining the operations to do in a training step
def train_step_one(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer):
    # Put model in train mode
    model.train()
    # Setup train loss and train accuracy values
    train_loss,train_acc = 0,0
    # Loop through DataLoader batches
    for batch,sample_batched in enumerate(dataloader):
        # Send data to target device
        X, y = sample_batched
        X.requires_grad = True
        # Forward pass
        y_pred = model(X)

        # Calculate  and accumulate loss
        loss = loss_fn(y_pred,y)
        train_loss += loss.item()
        # Optimizer zero grad
        optimizer.zero_grad()
        # Loss backward
        loss.backward()
        # Optimizer step
        optimizer.step()
        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = F.sigmoid(y_pred).round()
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss,train_acc
from torchmetrics.classification import BinaryConfusionMatrix
from sklearn.metrics import f1_score
# Defining the operations to do in a validation step
def val_step_one(model:torch.nn.Module,
              dataloader:torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module):
    # Put model in eval mode
    model.eval()
    bcm = BinaryConfusionMatrix()
    # Setup validation loss and validation accuracy values
    val_loss,val_acc = 0,0
    y_pred = torch.IntTensor()
    y_true = torch.IntTensor()
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch,sample_batched in enumerate(dataloader):
            # Send data to target device
            X,y = sample_batched
            
            # Forward pass
            val_pred_logits = model(X)
            # Calculate and accumulate loss
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()
            # Calculate and accumulate accuracy
            old_res = val_pred_logits.argmax(1)
            val_pred_labels = F.sigmoid(val_pred_logits).round()
            val_acc += ((val_pred_labels == y).sum().item()/len(val_pred_labels))
            y_pred=torch.cat((y_pred,val_pred_labels.cpu().detach()))
            y_true=torch.cat((y_true,y.cpu().detach()))
    # Adjust metrics to get average loss and accuracy per batch
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    f1 = f1_score(y_true.flatten(), y_pred.flatten())
    conf_mat = bcm(y_pred.flatten(), y_true.flatten())
    return val_loss,val_acc, f1, conf_mat

# Taking in various parameters required for training and validation steps
def train_one(model:torch.nn.Module,
          train_dataloader:torch.utils.data.DataLoader,
          val_dataloader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          loss_fn:torch.nn.Module = nn.CrossEntropyLoss(),
          epochs:int = 5,
          split:int = 0,
          dir = ""):
    # Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    # Instantiating the best validation accuracy
    best_val = 0
    # Loop through training and validation steps for a number of epochs
    for epoch in range(epochs):
        train_loss, train_acc = train_step_one(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        val_loss, val_acc, _, _ = val_step_one(model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn)
        # Saving the model obtaining the best validation accuracy through the epochs
        if val_acc > best_val:
            best_val = val_acc
            checkpoint = {"model": model,
                          "state_dict": model.state_dict(),
                          "optimizer": optimizer.state_dict()}
            checkpoint_name = f"{dir}/One_checkpoint_"+str(split)+".pth"
            torch.save(checkpoint, checkpoint_name)
        # else:
        #     continue
        # Print out what's happening
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc:.4f}"
            )
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
    # Return the filled results at the end of the epochs
    return results