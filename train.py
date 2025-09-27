import torch
from neural_solver import NeuralSolver, TileCNN 

# ========= batching & collation =========

def pad_and_stack(tensors, pad_dim, pad_value=0.0, dtype=None):
    """Pad a list of tensors along dim=pad_dim to the max size, then stack (adds batch dim)."""
    max_len = max(t.shape[pad_dim] for t in tensors)
    out = []
    for t in tensors:
        if dtype is not None:
            t = t.to(dtype)
        pad_sizes = [0,0] * t.dim()
        pad_amt = max_len - t.shape[pad_dim]
        pad_sizes[-2*pad_dim-1] = pad_amt  # torch.nn.functional.pad uses reverse order
        out.append(F.pad(t, pad=pad_sizes, mode="constant", value=pad_value))
    return torch.stack(out, dim=0), max_len

def collate_step_batches(batches: list[StepBatch]) -> StepBatch:
    """Collate a list of StepBatch (variable M/A) into a single batched StepBatch with masks."""
    # occ: (C,H,W) → stack directly (H/W vary lightly if you keep same crop policy)
    occ = torch.stack([b.occ.squeeze(0) for b in batches], dim=0)

    # tiles_left: (M_i, d) with mask (M_i,)
    tiles, Ms = [], []
    masks = []
    for b in batches:
        tiles.append(b.tiles_left.squeeze(0))
        masks.append(b.tiles_left_mask.squeeze(0))
        Ms.append(b.tiles_left.size(1))
    tiles_left, Mmax = pad_and_stack(tiles, pad_dim=0, pad_value=0.0)
    tiles_left_mask, _ = pad_and_stack(masks, pad_dim=0, pad_value=0.0, dtype=torch.bool)

    # candidates: (A_i, F), mask (A_i,), tile_idx (A_i,)
    cfeats, cmasks, cidxs, As = [], [], [], []
    for b in batches:
        cfeats.append(b.cand_feats.squeeze(0))
        cmasks.append(b.cand_mask.squeeze(0))
        cidxs.append(b.cand_tile_idx.squeeze(0))
        As.append(b.cand_feats.size(1))
    cand_feats, Amax = pad_and_stack(cfeats, pad_dim=0, pad_value=0.0)
    cand_mask, _     = pad_and_stack(cmasks, pad_dim=0, pad_value=0.0, dtype=torch.bool)
    cand_tile_idx, _ = pad_and_stack(cidxs, pad_dim=0, pad_value=0, dtype=torch.long)

    # expert actions (optional)
    if all(b.expert_action is not None for b in batches):
        expert_action = torch.stack([b.expert_action for b in batches], dim=0)
    else:
        expert_action = None

    return StepBatch(
        occ=occ, tiles_left=tiles_left, tiles_left_mask=tiles_left_mask,
        cand_feats=cand_feats, cand_mask=cand_mask, cand_tile_idx=cand_tile_idx,
        expert_action=expert_action
    )

# ========= environment pool =========

class EnvPool:
    """Manages N parallel single-instance envs. Simple Python loop (no subprocess)."""
    def __init__(self, N: int, T: int, n: int, alphabet: int, seed: int = 0):
        self.N = N
        self.T = T
        self.n = n
        self.alphabet = alphabet
        rng = np.random.RandomState(seed)
        self.envs: list[TilePlacementEnv] = []
        self.tiles_list: list[list[np.ndarray]] = []
        for i in range(N):
            tiles = make_synthetic_tiles(T, n, alphabet, seed=rng.randint(0, 1<<31))
            self.tiles_list.append(tiles)
            self.envs.append(TilePlacementEnv(tiles, alphabet))

    def reset_all(self):
        for env in self.envs:
            env.reset()

    def done_mask(self) -> np.ndarray:
        return np.array([env.done for env in self.envs], dtype=bool)

    def step_with_actions(self, actions: list[int]):
        """Apply action index per env (assumes you just scored candidates)."""
        for i, env in enumerate(self.envs):
            if env.done: 
                continue
            raw_cands = env.generate_candidates()
            if not raw_cands:
                # dead-end; mark as done
                env.done = True
                continue
            a = int(actions[i])
            a = max(0, min(a, len(raw_cands)-1))
            env.step(raw_cands[a])

    def build_batches(self, tile_emb_tables: list[torch.Tensor], raster_pad=1, max_crop_hw=None) -> list[StepBatch]:
        """Return a list of StepBatch (B=1 each), one per env not done."""
        out = []
        for env, tile_embs in zip(self.envs, tile_emb_tables):
            if env.done:
                continue
            sb = build_step_batch_from_env(env, tile_embs, raster_pad=raster_pad, max_crop_hw=max_crop_hw)
            out.append(sb)
        return out

# ========= multi-env IL training =========

def train_imitation_multi(
    model: NeuralSolver,
    tile_cnn: TileCNN,
    pool: EnvPool,
    device: torch.device,
    epochs: int = 5,
    steps_per_epoch: int = 2000,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    print_every: int = 200,
):
    # precompute tile embeddings per env once (they're static for synthetic tiles)
    with torch.no_grad():
        tile_emb_tables = []
        for tiles in pool.tiles_list:
            tile_emb_tables.append(precompute_tile_embeddings(tiles, pool.alphabet, tile_cnn.to(device), device).to(device))

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for ep in range(1, epochs+1):
        pool.reset_all()
        running = {"nll":0.0, "count":0, "avg_m":0.0, "placed":0.0}

        for step in range(1, steps_per_epoch+1):
            # build step batches for all active envs
            batches = pool.build_batches(tile_emb_tables)
            if not batches:
                # all done; restart
                pool.reset_all()
                continue

            # greedy expert for supervision
            for b in batches:
                b.expert_action = choose_expert_action(b)

            batch = collate_step_batches(batches)
            logits, _ = model(
                batch.occ.to(device),
                batch.tiles_left.to(device),
                batch.tiles_left_mask.to(device),
                batch.cand_feats.to(device),
                batch.cand_mask.to(device),
                batch.cand_tile_idx.to(device),
            )
            loss = imitation_loss(logits, batch.cand_mask.to(device), batch.expert_action.to(device))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            running["nll"] += float(loss.item())
            running["count"] += 1

            # act with model argmax on each env
            with torch.no_grad():
                probs = masked_softmax(logits, batch.cand_mask.to(device), dim=-1).cpu()
                acts = probs.argmax(dim=-1).tolist()
            pool.step_with_actions(acts)

            if step % print_every == 0:
                # quick metrics from a subset
                avg_nll = running["nll"] / max(1, running["count"])
                # sample 4 envs to compute current m and placed
                ms, placed = [], []
                for env in pool.envs[:min(4, pool.N)]:
                    m,_ = layout_bbox(env.placements, env.n)
                    ms.append(m); placed.append(len(env.placed_ids))
                print(f"[ep {ep} step {step}] nll={avg_nll:.4f}  m~{np.mean(ms):.2f}  placed~{np.mean(placed):.1f}/{pool.T}")

        # epoch-end: quick full evaluation
        eval_stats = evaluate_pool(model, pool, tile_emb_tables, device)
        print(f"[ep {ep}] eval: best_m_avg={eval_stats['m_avg']:.2f}, complete%={100*eval_stats['complete_rate']:.1f}")

# ========= evaluation =========

@torch.no_grad()
def evaluate_pool(
    model: NeuralSolver,
    pool: EnvPool,
    tile_emb_tables: list[torch.Tensor],
    device: torch.device,
    beam_width: int = 1,  # 1 = greedy
):
    model.eval()
    # fresh reset for eval
    pool.reset_all()
    done = pool.done_mask()
    while not done.all():
        batches = pool.build_batches(tile_emb_tables)
        if not batches:
            break
        batch = collate_step_batches(batches)
        logits, _ = model(
            batch.occ.to(device),
            batch.tiles_left.to(device),
            batch.tiles_left_mask.to(device),
            batch.cand_feats.to(device),
            batch.cand_mask.to(device),
            batch.cand_tile_idx.to(device),
        )
        if beam_width <= 1:
            probs = masked_softmax(logits, batch.cand_mask.to(device), dim=-1).cpu()
            acts = probs.argmax(dim=-1).tolist()
        else:
            # naive per-step top-k (doesn't maintain per-branch env copies to full depth)
            probs = masked_softmax(logits, batch.cand_mask.to(device), dim=-1).cpu()
            topv, topi = probs.topk(min(beam_width, probs.size(-1)), dim=-1)
            acts = topi[:, 0].tolist()  # use best for now; expand to full beam if desired
        pool.step_with_actions(acts)
        done = pool.done_mask()

    # collect stats
    ms, comp = [], 0
    for env in pool.envs:
        m,_ = layout_bbox(env.placements, env.n)
        ms.append(m)
        if env.done and len(env.placed_ids) == pool.T:
            comp += 1
    return {"m_avg": float(np.mean(ms)), "complete_rate": comp / pool.N}

# ========= main wiring =========

def run_training_main(
    N_pool=32, T=24, n=6, alphabet=4, epochs=5, steps_per_epoch=2000, lr=1e-3, device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model sizes
    c_occ = alphabet + 1
    d_tile = 64
    d_model = 128
    cand_feat_dim = 10  # as defined in build_step_batch_from_env

    # instantiate
    tile_cnn = TileCNN(in_ch=alphabet, d_tile=d_tile).to(device)
    model = NeuralSolver(c_occ=c_occ, d_tile=d_tile, d_model=d_model, cand_feat_dim=cand_feat_dim).to(device)

    # pool
    pool = EnvPool(N=N_pool, T=T, n=n, alphabet=alphabet, seed=0)

    # train
    train_imitation_multi(
        model=model, tile_cnn=tile_cnn, pool=pool, device=device,
        epochs=epochs, steps_per_epoch=steps_per_epoch, lr=lr, grad_clip=1.0, print_every=200
    )

    # final eval
    with torch.no_grad():
        tile_emb_tables = []
        for tiles in pool.tiles_list:
            tile_emb_tables.append(precompute_tile_embeddings(tiles, alphabet, tile_cnn, device).to(device))
    eval_stats = evaluate_pool(model, pool, tile_emb_tables, device, beam_width=1)
    print("Final eval:", eval_stats)
    return model

# If running as script:
if __name__ == "__main__":
    run_training_main()
