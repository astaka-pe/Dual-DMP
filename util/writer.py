import os
import time

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    raise Exception('Tensorboard is not found in PyTorch module. Update its version higher than 1.5.0')


class Writer:
    def __init__(self, opt):
        self.name = opt.name
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.log_name = os.path.join(self.save_dir, 'loss_log.txt')
        self.testacc_log = os.path.join(self.save_dir, 'testacc_log.txt')
        self.start_logs()
        self.nexamples = 0
        self.ncorrect = 0

        if opt.is_train and not opt.no_vis:
            self.display = SummaryWriter(comment=opt.name)
        else:
            self.display = None

    def start_logs(self):
        """ creates test / train log files """
        if self.opt.is_train:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)
        else:
            with open(self.testacc_log, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Testing Acc (%s) ================\n' % now)

    def print_current_losses(self, epoch, i, losses, t, t_data):
        """ prints train loss to terminal / file """
        message = '(epoch: %d, iters: %d, time: %.5f, data: %.5f) loss: %.5e ' \
                  % (epoch, i, t, t_data, losses.item())
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_loss(self, loss, epoch, i, n):
        iters = i + (epoch - 1) * n
        if self.display:
            self.display.add_scalar('data/train_loss', loss, iters)

    def plot_loss_with_label(self, loss, label, epoch, i, n):
        iters = i + (epoch - 1) * n
        if self.display:
            self.display.add_scalar(label, loss, iters)

    def plot_model_wts(self, model, epoch):
        if self.opt.is_train and self.display:
            for name, param in model.net.named_parameters():
                self.display.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    def plot_mesh(self, verts, faces, colors=None, epoch=0, label='data/test'):
        if self.display is not None:
            self.display.add_mesh(label, verts, colors=colors, faces=faces, global_step=epoch)

    def print_acc(self, epoch, acc):
        """ prints test accuracy to terminal / file """
        message = 'epoch: {}, TEST ACC: [{:.5} %]\n' \
            .format(epoch, acc * 100)
        print(message)
        with open(self.testacc_log, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_acc(self, acc, epoch):
        if self.display is not None:
            self.display.add_scalar('data/test_acc', acc, epoch)

    def reset_counter(self):
        """
        counts # of correct examples
        """
        self.ncorrect = 0
        self.nexamples = 0

    def update_counter(self, ncorrect, nexamples):
        self.ncorrect += ncorrect
        self.nexamples += nexamples

    @property
    def acc(self):
        return float(self.ncorrect) / self.nexamples

    def flush(self):
        if self.display is not None:
            self.display.flush()

    def close(self):
        if self.display is not None:
            self.display.close()
