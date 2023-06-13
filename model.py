import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return (x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma,
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # 1, x
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)

            done = (done, )

        print('State: ', 'DS, DR, DL, L, R, U, D, FL, FR, FD, FU')
        print('State: ', state)
        print('Next State: ', next_state)
        print('Action: ', action)
        print('Reward: ', reward)
        print('Done: ', done)

        # 1: predicted Q values with current state
        pred = self.model(state)

        print('Q0_pred: ', pred)

        target = pred.clone()

        for idx in range(len(done)):
            print('idx ', idx, ' of ', range(len(done)))
            Q_new = reward[idx]
            if not done[idx]:
                print('Not done')
                print('Q1_pred: ', self.model(next_state[idx]))
                print('Gamma = ', self.gamma)
                print('Reward = ', reward[idx])
                Q_new = reward[idx] + torch.mul(
                    torch.max(self.model(next_state[idx])), self.gamma[0])
                print('Q1_max = reward + gamma * max(Q) = ', Q_new)

            target[idx][torch.argmax(action).item()] = Q_new
            print('Target: ', target)

        # zero_grad clears gradients from previous steps, otherwise they'll
        # accumulate. We only want to solve for the current step's cost
        self.optimizer.zero_grad()
        # calculate the loss for this step
        loss = self.criterion(target, pred)
        # calculate the gradients of the loss using backprop
        loss.backward()
        # optimizer takes a step based on the gradients
        self.optimizer.step()

        # input("Press Enter to continue...")
