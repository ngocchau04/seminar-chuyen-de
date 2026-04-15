# Bao cao tom luoc do an Transformer Encoder

## 1) Muc tieu va pham vi
Do an cai dat mot **Transformer Encoder don gian** cho bai toan phan loai cam xuc 3 lop (negative/neutral/positive) bang PyTorch.

Yeu cau ky thuat bat buoc:
- Scaled Dot-Product Attention
- Self-Attention
- FeedForwardNetwork (2 lop)
- TransformerEncoderBlock (Add&Norm + FFN + Add&Norm)

## 2) Nhung gi da lam duoc
### 2.1 Cai dat model (dat)
Trong `model.py`:
- `scaled_dot_product_attention(Q, K, V)` da cai dat dung cong thuc:
  - `scores = Q @ K^T / sqrt(d_k)`
  - `weights = softmax(scores, dim=-1)`
  - `output = weights @ V`
- `SelfAttention`: dung 3 phep chieu `q_proj`, `k_proj`, `v_proj`.
- `FeedForwardNetwork`: `Linear(d_model, d_ff) -> ReLU -> Linear(d_ff, d_model)`.
- `TransformerEncoderBlock`: dung residual + `LayerNorm` 2 lan theo dung thu tu.

### 2.2 Kiem thu model (dat)
Da chay thanh cong:
- `python model.py`

Ket qua:
- scaled_dot_product_attention: PASSED
- SelfAttention: PASSED
- FeedForwardNetwork: PASSED
- TransformerEncoderBlock: PASSED

### 2.3 Huan luyen va so sanh cau hinh (dat)
Da chay thanh cong:
1. `python train.py`
2. `python train.py --run_all`
3. `python train.py --d_model 128 --d_ff 256`

Ket qua (tu `python train.py --run_all`):

| Model | Train Acc | Val Acc | Test Acc | Final Train Loss |
|---|---:|---:|---:|---:|
| Transformer_d64_ff128 | 0.9905 | 0.9556 | 0.9778 | 0.0390 |
| Transformer_d128_ff256 | 0.9976 | 0.9667 | 0.9778 | 0.0091 |
| Transformer_d32_ff64 | 0.9167 | 0.8889 | 0.8444 | 0.1915 |
| MLPBaseline_d64 | 0.8762 | 0.7667 | 0.8111 | 0.5321 |

Nhan xet nhanh:
- Transformer vuot baseline MLP ro ret.
- `d128_ff256` va `d64_ff128` cho test acc cao nhat (0.9778).

### 2.4 Visualize attention (dat)
Da chay thanh cong:
1. `python visualize.py`
2. `python visualize.py --sentence "this film is absolutely terrible"`

Da tao/sao luu 3 heatmap de phan tich:
- `results/attention_heatmap_case1_negative.png`
- `results/attention_heatmap_case2_neutral.png`
- `results/attention_heatmap_case3_positive.png`

## 3) Nhung gi chua lam xong / con thieu
- Chua co file bao cao PDF 6-10 trang theo muc 6 cua de.
- Chua dong goi file `.zip` nop bai theo cau truc muc 8.
- Chua viet day du phan "Error Analysis" 5-10 mau sai trong test set (hien moi co khung ket qua va visual).

## 4) Phan tich Attention (toi thieu 3 cau)
Model dung de visualize/phan tich: `results/model_Transformer_d128_ff256.pt`.

### Case 1 (negative)
- Cau: **"this film is absolutely terrible"**
- Du doan: **negative**
- Heatmap: `results/attention_heatmap_case1_negative.png`
- So lieu attention (trich tu tensor):
  - Global focus token: `terrible` (mean attention ~ **0.9881**)
  - Nhieu query token (`this`, `film`, `is`, `absolutely`) deu tap trung vao `terrible` (weight ~0.97-0.99)
- Nhan xet:
  - Mo hinh nhan dien dung tu cam xuc manh (`terrible`) va cho trong so rat cao.
  - Kieu tap trung nay hop ly voi bai toan sentiment.

### Case 2 (neutral)
- Cau: **"we discussed the movie in class at home"**
- Du doan: **neutral**
- Heatmap: `results/attention_heatmap_case2_neutral.png`
- So lieu attention:
  - Global focus token: `class` (mean attention ~ **0.3937**)
  - Cac token chia chu y qua lai giua `in` va `class`, khong co 1 tu cam xuc chi phoi.
- Nhan xet:
  - Pattern attention phan tan, thien ve boi canh/su kien hon la tu mang cam xuc.
  - Du doan neutral la hop ly.

### Case 3 (positive)
- Cau: **"this movie is absolutely wonderful"**
- Du doan: **positive**
- Heatmap: `results/attention_heatmap_case3_positive.png`
- So lieu attention:
  - Global focus token: `wonderful` (mean attention ~ **0.9583**)
  - Da so query token deu tap trung vao `wonderful` (weight cao ~0.93-0.99)
- Nhan xet:
  - Tu cam xuc manh tich cuc (`wonderful`) dong vai tro trung tam.
  - Hanh vi attention doi xung voi case negative (`terrible`), cho thay mo hinh hoc duoc tu khoa cam xuc.

## 5) Danh gia overfitting ngan gon
- O cau hinh lon (`d128_ff256`), train acc rat cao (gan 1.0), trong khi val/test khong tang tuong ung o cuoi epoch.
- Dau hieu overfit nhe xuat hien sau giai doan val dat dinh (val loss/val acc bat dau dao dong).
- Tuy nhien test acc van cao (0.9778), nen muc overfit hien tai chua gay giam ket qua test ro ret.

## 6) Ket luan tam thoi
- Phan implementation va pipeline chay da dap ung phan lon yeu cau ky thuat cua de bai.
- Can hoan tat 2 hang muc nop bai con thieu:
  1. Bao cao PDF day du theo cau truc muc 6.
  2. Dong goi ZIP dung cau truc muc 8.
