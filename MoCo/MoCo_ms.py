import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter, ms_function
from mindspore.communication import get_rank


class QueueDict(nn.Cell):
    def __init__(self, dim=128, K=65536):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        """
        super(QueueDict, self).__init__()
        self.K = K

        # create the queue
        self.queue = Parameter(ops.StandardNormal()((dim, K)), name="queue", requires_grad=False)
        self.queue = ops.L2Normalize(axis=0)(self.queue)

    @ms_function
    def construct(self, keys, ptr):
        # gather keys before updating queue
        keys = ops.AllGather()(keys)
        
        batch_size = keys.shape[0]
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        
        return ptr


class BatchShuffle(nn.Cell):
    def __init__(self):
        super(BatchShuffle, self).__init__()

    @ms_function
    def construct(self, x, gpu_idx):
        """ Batch shuffle, for making use of BatchNorm. """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = ops.AllGather()(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        n = Tensor([batch_size_all], dtype=ms.int32)
        idx_shuffle = ops.Randperm(max_length=batch_size_all)(n)

        # broadcast to all gpus
        ops.Broadcast(0)((idx_shuffle,))

        # index for restoring
        _, idx_unshuffle = ops.Sort()(idx_shuffle.astype(ms.float32))

        # shuffled index for this gpu
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle


class BatchUnshuffle(nn.Cell):
    def __init__(self):
        super(BatchUnshuffle, self).__init__()

    @ms_function
    def construct(self, x, idx_unshuffle, gpu_idx):
        """ Undo batch shuffle. """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = ops.AllGather()(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

 
class MoCo(nn.Cell):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, args=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # class_num is the output fc dimension
        self.encoder_q = base_encoder(class_num=dim)
        self.encoder_k = base_encoder(class_num=dim)

        # print('Args: args.hidden', args.hidden)
        if mlp and args.hidden:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.end_point.in_channels
            if args.hidden != -1:
                hidden_mlp = args.hidden
            else: hidden_mlp = dim_mlp
            self.encoder_q.end_point = nn.SequentialCell([nn.Dense(dim_mlp, hidden_mlp), nn.ReLU(), nn.Dense(hidden_mlp, dim)])
            self.encoder_k.end_point = nn.SequentialCell([nn.Dense(dim_mlp, hidden_mlp), nn.ReLU(), nn.Dense(hidden_mlp, dim)])

        for param_q, param_k in zip(self.encoder_q.get_parameters(), 
                                    self.encoder_k.get_parameters()):
            param_k = param_q.clone()  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.QueueDict = QueueDict(dim, K)
        self.queue_ptr = Parameter(ops.Zeros()(1, ms.int32), name="queue_ptr", requires_grad=False)

    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.get_parameters(), 
                                    self.encoder_k.get_parameters()):
            param_k.set_data(param_k.data * self.m + param_q.data * (1 - self.m))
        
    def construct(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = ops.L2Normalize(axis=1)(q)

        # compute key features
        self._momentum_update_key_encoder()  # update the key encoder

        # shuffle for making use of BN
        gpu_idx = get_rank()
        # im_k, idx_unshuffle = BatchShuffle()(im_k, gpu_idx)

        k = self.encoder_k(im_k)  # keys: NxC
        k = ops.L2Normalize(axis=1)(k)

        # undo shuffle
        # k = BatchUnshuffle()(k, idx_unshuffle, gpu_idx)
        k = ops.stop_gradient(k)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = ops.Einsum('nc,nc->n')((q, k)).expand_dims(-1)
        # negative logits: NxK
        l_neg = ops.Einsum('nc,ck->nk')((q, self.QueueDict.queue.copy()))

        # logits: Nx(1+K)
        logits = ops.Concat(axis=1)([l_pos, l_neg])

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = ops.Zeros()((logits.shape[0], ), ms.int32)

        # dequeue and enqueue
        ptr = int(self.queue_ptr)
        ptr = self.QueueDict(k, ptr)
        self.queue_ptr[0] = ptr

        return logits, labels