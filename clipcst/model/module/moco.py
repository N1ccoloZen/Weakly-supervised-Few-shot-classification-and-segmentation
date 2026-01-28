import torch
import torch.nn.functional as F
import torch.distributed as dist

class ClassCondMoCoQueues():
    '''
    Class-conditioned memory bank (one 'circular' queue per class)
    '''
    def __init__(self, num_classes, embed_dim, k_per_class=256, temperature=0.07, device='cuda'):
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.K = k_per_class
        self.temp = temperature
        self.device = device

        q = torch.randn(num_classes, k_per_class, embed_dim, dtype=torch.float32)
        q = F.normalize(q, dim=-1)

        self.queues = q.to(device)
        self.ptrs = torch.zeros(num_classes, dtype=torch.long, device=device)

    def _all_gather(self, x):
        '''
        All gather and concatenate
        '''
        if not dist.is_initialized() or not dist.is_available():
            return x
        world_size = dist.get_world_size()

        gather_list = [torch.zeros_like(x) for _ in range(world_size)]

        dist.all_gather(gather_list, x.contiguous())
        return torch.cat(gather_list, dim=0)
    
    @torch.no_grad()
    def enqueue_from_all_ranks(self, class_ids_local, keys_local):
        '''gathers keys from all ranks and enqueues them into class queues deterministically (one queue for each class)'''

        class_ids_local = class_ids_local.to(self.device, dtype=torch.long)
        keys_local = keys_local.contiguous()

        class_ids_all = self._all_gather(class_ids_local)
        keys_all = self._all_gather(keys_local)

        B_all = class_ids_all.shape[0]

        #enqueue in order
        for idx in range(B_all):
            c = int(class_ids_all[idx].item())
            k = keys_all[idx]

            ptr = int(self.ptrs[c].item())
            self.queues[c, ptr] = k.to(self.device)
            self.ptrs[c] = (ptr + 1) % self.K

    def get_negatives_for_class(self, class_id):
        '''returns neg_keys by concatenating queues for all classes != class_ids'''

        mask = torch.arange(self.num_classes, device=self.device) != class_id

        negs = self.queues[mask].reshape(-1, self.embed_dim)
 
        return negs
    
    def compute_contrastive(self, qry_emb, key_emb, class_ids, gt_presence):
        '''returns mean loss over local batch'''
        device = qry_emb.device

        B = qry_emb.shape[0]
        losses = []

        for i in range(B):
            q = qry_emb[i]
            c = int(class_ids[i].item())
            present = bool(gt_presence[i].item())

            neg_keys = self.get_negatives_for_class(c)
            neg_keys = neg_keys.detach()

            if present:
                pos = key_emb[i].unsqueeze(0)

                logits_pos = (q @ pos.T) / self.temp
                logits_neg = (q @ neg_keys.T) / self.temp
                logits = torch.cat([logits_pos ,logits_neg], dim=1)

                labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
                loss = F.binary_cross_entropy_with_logits(logits, labels)

            else:
                sims = (q @ neg_keys.T)

                loss = F.softplus(sims).mean()

            losses.append(loss)

        return torch.stack(losses).mean() 