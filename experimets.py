import torch
try:
    checkpoint = torch.load("checkpoint.pt")
except:
    checkpoint = {"epoch":0,
                "loss":999}
    torch.save(checkpoint, "checkpoint.pt")