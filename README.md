Code for our Paper 
Alleviating Dimensional Collapse Problem in Deep Recommender Models by Designing Uniformity Layers

# Run (Equlation 6: LightGCN on ml-100k dataset), Equlation=7 is for equlation 7 in this paper
```
python run_recbole.py -m LightGCN -d ml-100k --n_layers=2 --gpu_id=0 --ULRec=Yes --Equlation=6 --alpha=0.9 
```

# result:
('recall@5', 0.1704), ('recall@10', 0.2646), ('recall@20', 0.3857), ('mrr@5', 0.4952), ('mrr@10', 0.5124), ('mrr@20', 0.5188), ('ndcg@5', 0.311), ('ndcg@10', 0.3115), ('ndcg@20', 0.3298)])
