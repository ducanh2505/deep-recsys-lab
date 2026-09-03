# LightGCN: Công thức toán học, mô hình và hàm mất mát

> **Paper:** *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*
> Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang — SIGIR 2020.
> arXiv: https://arxiv.org/abs/2002.02126

---

## 1. Ý tưởng chính

LightGCN được xây dựng cho bài toán **Collaborative Filtering (CF)** trên đồ thị hai phía user–item.

Điểm xuất phát của paper là quan sát rằng các GCN trước đó cho recommendation, đặc biệt là NGCF, sử dụng nhiều thành phần vốn phổ biến trong GCN:

- feature transformation,
- nonlinear activation,
- message passing,
- self-connection,
- các phép tương tác feature phức tạp.

Tuy nhiên, user và item trong collaborative filtering thường **không có feature giàu ngữ nghĩa**. Mỗi node về bản chất chỉ được nhận diện bởi ID và được ánh xạ sang một latent embedding.

LightGCN vì vậy loại bỏ:

$$
\boxed{\text{Feature Transformation}}
$$

và

$$
\boxed{\text{Nonlinear Activation}}
$$

chỉ giữ lại:

$$
\boxed{\text{Neighborhood Aggregation}}
$$

Có thể tóm tắt mô hình bằng một câu:

$$
\boxed{
\text{LightGCN}
=
\text{Trainable ID Embeddings}
+
\text{Linear Graph Propagation}
+
\text{Layer Combination}
}
$$

---

# 2. User–Item Interaction Graph

Giả sử:

- $U$: tập users,
- $I$: tập items,
- $E$: tập các interaction quan sát được.

Ta xây dựng một bipartite graph:

$$
G=(U,I,E)
$$

Trong đó:

$$
(u,i)\in E
$$

có nghĩa user $u$ đã tương tác với item $i$.

Ví dụ:

```text
User 1 ───── Item A
   │
   └──────── Item B ───── User 2
                           │
                           └──── Item C
```

Với implicit feedback, interaction có thể là:

- click,
- purchase,
- watch,
- listen,
- like,
- check-in,
- v.v.

---

# 3. Interaction Matrix

Gọi:

$$
R\in\mathbb{R}^{M\times N}
$$

với:

- $M=|U|$,
- $N=|I|$.

Ta có:

$$
R_{ui}
=
\begin{cases}
1, & \text{nếu user }u\text{ đã tương tác với item }i\\
0, & \text{ngược lại}
\end{cases}
$$

Từ đó adjacency matrix của user–item graph có dạng:

$$
A=
\begin{bmatrix}
0 & R\\
R^T & 0
\end{bmatrix}
$$

Kích thước:

$$
A\in\mathbb{R}^{(M+N)\times(M+N)}
$$

Do graph là bipartite:

- user chỉ nối với item,
- item chỉ nối với user,
- không có cạnh user–user trực tiếp,
- không có cạnh item–item trực tiếp.

---

# 4. Initial Embeddings

LightGCN học một embedding cho mỗi user và item:

$$
e_u^{(0)}\in\mathbb{R}^{d}
$$

$$
e_i^{(0)}\in\mathbb{R}^{d}
$$

Trong đó $d$ là embedding dimension.

Toàn bộ user embeddings:

$$
E_U^{(0)}
\in
\mathbb{R}^{M\times d}
$$

Toàn bộ item embeddings:

$$
E_I^{(0)}
\in
\mathbb{R}^{N\times d}
$$

Ghép lại:

$$
E^{(0)}
=
\begin{bmatrix}
E_U^{(0)}\\
E_I^{(0)}
\end{bmatrix}
$$

với:

$$
E^{(0)}
\in
\mathbb{R}^{(M+N)\times d}
$$

## 4.1 Initial embedding có phải one-hot không?

Không.

Paper nói rằng user và item về cơ bản chỉ được **mô tả bởi ID**, có thể hiểu ở mức input là one-hot identifier. Tuy nhiên embedding được propagate trong LightGCN là **dense trainable embedding**:

$$
e_u^{(0)}
$$

chứ không phải one-hot vector.

Có thể hình dung:

$$
\text{User ID}
\rightarrow
\text{Embedding Lookup}
\rightarrow
e_u^{(0)}
$$

Nếu biểu diễn ID bằng one-hot:

$$
x_u\in\mathbb{R}^{M}
$$

và embedding table là:

$$
W_U\in\mathbb{R}^{M\times d}
$$

thì:

$$
e_u^{(0)}=x_u^TW_U
$$

Do $x_u$ là one-hot, phép nhân này chỉ tương đương với việc lấy một row của embedding table:

$$
e_u^{(0)}=W_U[u]
$$

Do đó one-hot chỉ là cách biểu diễn ID về mặt khái niệm; parameter thực sự được train là dense embedding.

---

# 5. Từ NGCF đến LightGCN

Một GCN layer thông thường có thể viết gần đúng:

$$
E^{(k+1)}
=
\sigma
\left(
\tilde A E^{(k)}W^{(k)}
\right)
$$

Trong đó:

- $\tilde A$: normalized adjacency matrix,
- $W^{(k)}$: learnable feature transformation,
- $\sigma(\cdot)$: nonlinear activation.

LightGCN loại bỏ:

$$
W^{(k)}
$$

và:

$$
\sigma(\cdot)
$$

nên layer chỉ còn:

$$
\boxed{
E^{(k+1)}
=
\tilde A E^{(k)}
}
$$

Đây là công thức cốt lõi của LightGCN.

---

# 6. Degree Normalization

Gọi degree matrix:

$$
D
$$

trong đó:

$$
D_{vv} = |\mathcal N_v|
$$

với $\mathcal N_v$ là tập hàng xóm của node $v$.

Normalized adjacency matrix:

$$
\boxed{
\tilde A
=
D^{-1/2}AD^{-1/2}
}
$$

Normalization này giúp kiểm soát ảnh hưởng của các node có degree lớn.

Nếu một item rất phổ biến, ví dụ được hàng triệu user tương tác, ta không muốn embedding của item đó có ảnh hưởng quá lớn lên tất cả users.

---

# 7. User-side Propagation

Tại layer $k+1$, embedding của user $u$ được tính từ embedding của các item mà user tương tác:

$$
\boxed{
e_u^{(k+1)}
=
\sum_{i\in\mathcal N_u}
\frac{1}
{\sqrt{|\mathcal N_u|}
\sqrt{|\mathcal N_i|}}
e_i^{(k)}
}
$$

Trong đó:

- $\mathcal N_u$: các item liên kết với user $u$,
- $\mathcal N_i$: các user liên kết với item $i$.

Hệ số:

$$
\frac{1}
{\sqrt{|\mathcal N_u|}
\sqrt{|\mathcal N_i|}}
$$

là symmetric normalization.

---

# 8. Item-side Propagation

Tương tự:

$$
\boxed{
e_i^{(k+1)}
=
\sum_{u\in\mathcal N_i}
\frac{1}
{\sqrt{|\mathcal N_i|}
\sqrt{|\mathcal N_u|}}
e_u^{(k)}
}
$$

Do graph là bipartite nên propagation luân phiên:

```text
Layer 0: User / Item embeddings
            │
Layer 1: User ← Item
         Item ← User
            │
Layer 2: User ← Item ← User
         Item ← User ← Item
            │
Layer 3: high-order collaborative signal
```

---

# 9. Ý nghĩa của từng layer

## Layer 0

$$
e_u^{(0)}
$$

là latent representation riêng của user.

Nó được học trực tiếp thông qua optimization.

---

## Layer 1

$$
e_u^{(1)}
$$

aggregate thông tin từ các item mà $u$ đã tương tác.

Ví dụ:

```text
Alice
 ├── Interstellar
 ├── Inception
 └── Tenet
```

thì:

$$
e_{\text{Alice}}^{(1)}
$$

là combination của embedding các phim này.

---

## Layer 2

Đường truyền:

$$
User
\rightarrow
Item
\rightarrow
User
$$

giúp user nhận thông tin từ các users có item interaction chung.

Ví dụ:

```text
Alice ───── Interstellar ───── Bob
Alice ───── Inception    ───── Bob
```

Alice và Bob không có edge trực tiếp nhưng có thể ảnh hưởng nhau sau hai propagation layers.

Đây chính là high-order collaborative signal.

---

## Layer $K$

Tổng quát:

$$
\boxed{
E^{(K)}
=
\tilde A^K E^{(0)}
}
$$

Do đó mỗi layer tương ứng với việc truyền thông tin thêm một hop trên graph.

---

# 10. Layer Combination

LightGCN không chỉ sử dụng embedding ở layer cuối.

Final embedding là weighted sum của tất cả các layer:

$$
\boxed{
e_u
=
\sum_{k=0}^{K}
\alpha_k e_u^{(k)}
}
$$

và:

$$
\boxed{
e_i
=
\sum_{k=0}^{K}
\alpha_k e_i^{(k)}
}
$$

Trong paper, một lựa chọn đơn giản là:

$$
\alpha_k
=
\frac{1}{K+1}
$$

Do đó nếu $K=3$:

$$
e_u
=
\frac{1}{4}
\left(
e_u^{(0)}
+
e_u^{(1)}
+
e_u^{(2)}
+
e_u^{(3)}
\right)
$$

Tương tự cho item.

## Tại sao cần combine nhiều layer?

Mỗi layer chứa một mức collaborative signal khác nhau:

| Layer | Ý nghĩa gần đúng |
|---|---|
| $0$ | latent preference riêng |
| $1$ | direct interaction |
| $2$ | similar users/items |
| $3+$ | high-order collaborative structure |

Nếu chỉ dùng layer cuối, representation có nguy cơ bị **over-smoothing**.

Việc giữ:

$$
E^{(0)}
$$

trong final representation cũng giúp bảo toàn thông tin riêng của từng node.

---

# 11. Matrix Form của toàn bộ LightGCN

Ta có:

$$
E^{(1)}
=
\tilde A E^{(0)}
$$

$$
E^{(2)}
=
\tilde A E^{(1)}
=
\tilde A^2E^{(0)}
$$

$$
E^{(3)}
=
\tilde A^3E^{(0)}
$$

Tổng quát:

$$
\boxed{
E^{(k)}
=
\tilde A^k E^{(0)}
}
$$

Final embedding:

$$
E
=
\sum_{k=0}^{K}
\alpha_k E^{(k)}
$$

Thay vào:

$$
\boxed{
E
=
\sum_{k=0}^{K}
\alpha_k
\tilde A^k
E^{(0)}
}
$$

Đặt:

$$
P(\tilde A)
=
\sum_{k=0}^{K}
\alpha_k\tilde A^k
$$

ta có:

$$
\boxed{
E
=
P(\tilde A)E^{(0)}
}
$$

Đây là một cách rất hữu ích để nhìn LightGCN:

> LightGCN thực hiện một phép **linear graph diffusion/filtering** lên initial latent embeddings.

---

# 12. Prediction Function

Sau khi thu được final user embedding:

$$
e_u
$$

và final item embedding:

$$
e_i
$$

prediction score được tính bằng inner product:

$$
\boxed{
\hat y_{ui}
=
e_u^T e_i
}
$$

Đây chính là cùng dạng prediction với Matrix Factorization.

Nếu:

$$
e_u^Te_i
$$

lớn thì model đánh giá user $u$ có khả năng quan tâm item $i$.

---

# 13. BPR — Bayesian Personalized Ranking Loss

LightGCN được train cho implicit feedback bằng **BPR loss**.

Với một triplet:

$$
(u,i,j)
$$

trong đó:

- $u$: user,
- $i$: positive item đã tương tác,
- $j$: negative item chưa quan sát tương tác.

Ta muốn:

$$
\hat y_{ui}
>
\hat y_{uj}
$$

Hay:

$$
\hat y_{ui}
-
\hat y_{uj}
>
0
$$

---

## 13.1 BPR Loss

BPR loss cho một triplet:

$$
\boxed{
\mathcal L_{BPR}
=
-
\log
\sigma
\left(
\hat y_{ui}
-
\hat y_{uj}
\right)
}
$$

với:

$$
\sigma(x)
=
\frac{1}{1+e^{-x}}
$$

Do:

$$
\hat y_{ui}
=
e_u^Te_i
$$

và:

$$
\hat y_{uj}
=
e_u^Te_j
$$

ta có:

$$
\boxed{
\mathcal L_{BPR}
=
-
\log
\sigma
\left(
e_u^Te_i
-
e_u^Te_j
\right)
}
$$

---

# 14. Trực giác của BPR

Giả sử:

$$
\hat y_{ui}=4
$$

và:

$$
\hat y_{uj}=1
$$

thì:

$$
\hat y_{ui}-\hat y_{uj}=3
$$

$$
\sigma(3)\approx0.953
$$

và loss nhỏ:

$$
-\log(0.953)
$$

Model được xem là ranking tốt.

Ngược lại nếu:

$$
\hat y_{ui}=1
$$

$$
\hat y_{uj}=4
$$

thì:

$$
\hat y_{ui}-\hat y_{uj}=-3
$$

$$
\sigma(-3)\approx0.047
$$

loss sẽ rất lớn.

Gradient sẽ đẩy model theo hướng:

$$
\hat y_{ui}\uparrow
$$

và:

$$
\hat y_{uj}\downarrow
$$

---

# 15. Regularization

Objective đầy đủ có thêm L2 regularization:

$$
\boxed{
\mathcal L
=
\mathcal L_{BPR}
+
\lambda
\|E^{(0)}\|_2^2
}
$$

Điểm đáng chú ý:

LightGCN chỉ cần regularize **initial embeddings**.

Lý do là:

$$
E^{(1)},
E^{(2)},...,E^{(K)}
$$

không phải các parameter độc lập.

Chúng đều được suy ra từ:

$$
E^{(0)}
$$

thông qua:

$$
E^{(k)}
=
\tilde A^kE^{(0)}
$$

Do đó tập trainable parameters về cơ bản là:

$$
\boxed{
\Theta
=
E^{(0)}
}
$$

---

# 16. Một bước training

Một mini-batch chứa các triplets:

$$
(u,i,j)
$$

### Bước 1 — Embedding lookup

Lấy:

$$
e_u^{(0)},
\quad
e_i^{(0)},
\quad
e_j^{(0)}
$$

---

### Bước 2 — Graph propagation

Tính toàn bộ:

$$
E^{(1)}
=
\tilde A E^{(0)}
$$

$$
E^{(2)}
=
\tilde A E^{(1)}
$$

...

$$
E^{(K)}
=
\tilde A E^{(K-1)}
$$

---

### Bước 3 — Layer combination

$$
E
=
\sum_{k=0}^{K}
\alpha_kE^{(k)}
$$

---

### Bước 4 — Prediction

$$
\hat y_{ui}
=
e_u^Te_i
$$

$$
\hat y_{uj}
=
e_u^Te_j
$$

---

### Bước 5 — BPR loss

$$
\mathcal L_{BPR}
=
-\log
\sigma(
\hat y_{ui}
-
\hat y_{uj}
)
$$

---

### Bước 6 — Regularization

$$
\mathcal L
=
\mathcal L_{BPR}
+
\lambda\|E^{(0)}\|^2
$$

---

### Bước 7 — Backpropagation

Gradient đi ngược từ loss qua:

```text
BPR loss
   ↓
final embeddings
   ↓
layer combination
   ↓
graph propagation
   ↓
E^(0)
```

Sau đó optimizer cập nhật:

$$
E^{(0)}
$$

---

# 17. Pseudocode

```python
# Trainable parameters
E_user = Embedding(num_users, dim)
E_item = Embedding(num_items, dim)

E0 = concat(E_user, E_item)

for epoch in range(num_epochs):

    # ----- Graph propagation -----

    embeddings = [E0]

    E = E0

    for k in range(K):
        E = normalized_adj @ E
        embeddings.append(E)

    # ----- Layer combination -----

    E_final = mean(embeddings, axis=0)

    user_final = E_final[:num_users]
    item_final = E_final[num_users:]

    # ----- BPR training -----

    for user, pos_item, neg_item in batches:

        u = user_final[user]
        i = item_final[pos_item]
        j = item_final[neg_item]

        pos_score = dot(u, i)
        neg_score = dot(u, j)

        bpr_loss = -log(sigmoid(pos_score - neg_score))

        reg_loss = lambda_reg * (
            ||E_user[user]||^2
            + ||E_item[pos_item]||^2
            + ||E_item[neg_item]||^2
        )

        loss = bpr_loss + reg_loss

        loss.backward()
        optimizer.step()
```

Đây là pseudocode khái niệm; implementation thực tế thường dùng sparse matrix multiplication để propagation hiệu quả hơn.

---

# 18. LightGCN vs Matrix Factorization

Matrix Factorization:

$$
\hat y_{ui}
=
(e_u^{(0)})^Te_i^{(0)}
$$

LightGCN:

$$
\hat y_{ui}
=
e_u^Te_i
$$

với:

$$
e_u
=
\sum_{k=0}^{K}
\alpha_ke_u^{(k)}
$$

Vì vậy có thể xem LightGCN như:

$$
\boxed{
\text{MF}
+
\text{Graph-based embedding propagation}
}
$$

Nếu:

$$
K=0
$$

thì không có graph propagation và mô hình về cơ bản trở về dạng Matrix Factorization.

---

# 19. LightGCN vs GCN thông thường

## GCN

$$
E^{(k+1)}
=
\sigma
\left(
\tilde AE^{(k)}W^{(k)}
\right)
$$

Có:

- graph aggregation,
- learnable transformation matrix,
- nonlinearity.

---

## LightGCN

$$
\boxed{
E^{(k+1)}
=
\tilde AE^{(k)}
}
$$

Không có:

$$
W^{(k)}
$$

Không có:

$$
\sigma
$$

Do đó LightGCN là một **linear propagation model**.

---

# 20. Tại sao bỏ nonlinear activation và feature transformation?

Trong node classification, input có thể là:

- text features,
- bag-of-words,
- image features,
- attributes.

Khi đó:

$$
W^{(k)}
$$

có thể học cách biến đổi feature space.

Trong collaborative filtering, input chủ yếu là:

$$
\text{user ID}
$$

và:

$$
\text{item ID}
$$

Embedding đã là latent parameter được học.

Paper cho thấy feature transformation và nonlinear activation không đem lại lợi ích rõ rệt trong setting này; ngược lại chúng có thể làm optimization khó hơn.

LightGCN vì vậy ưu tiên:

$$
\boxed{
\text{Propagate collaborative signal}
}
$$

thay vì:

$$
\boxed{
\text{Transform node features}
}
$$

---

# 21. Interpretation dưới góc nhìn graph filtering

Từ:

$$
E
=
\sum_{k=0}^{K}
\alpha_k\tilde A^k E^{(0)}
$$

có thể coi:

$$
\sum_{k=0}^{K}
\alpha_k\tilde A^k
$$

là một polynomial graph filter.

Graph structure quyết định cách latent information được khuếch tán.

Các node gần nhau trong interaction graph dần ảnh hưởng lên representation của nhau.

Điều này giải thích vì sao LightGCN có thể capture:

- direct preference,
- user similarity,
- item similarity,
- high-order collaborative relations.

---

# 22. Ví dụ nhỏ

Giả sử:

```text
U1 ─ I1
 │
 └── I2 ─ U2
          │
          └── I3
```

Initial embeddings:

$$
e_{U1}^{(0)},
e_{U2}^{(0)},
e_{I1}^{(0)},
e_{I2}^{(0)},
e_{I3}^{(0)}
$$

Sau layer 1:

$$
e_{U1}^{(1)}
\leftarrow
e_{I1}^{(0)},e_{I2}^{(0)}
$$

$$
e_{U2}^{(1)}
\leftarrow
e_{I2}^{(0)},e_{I3}^{(0)}
$$

Sau layer 2:

$$
e_{U1}^{(2)}
$$

có thể gián tiếp nhận signal từ:

$$
U2
$$

thông qua:

$$
U1
\rightarrow
I2
\rightarrow
U2
$$

Do đó LightGCN tự động khai thác user similarity thông qua graph mà không cần xây dựng user-user similarity matrix riêng.

---

# 23. BPR và Contrastive Learning Loss

BPR là loss gốc được dùng để optimize recommendation trong LightGCN.

### BPR

$$
\mathcal L_{BPR}
=
-\log\sigma(
s(u,i)-s(u,j)
)
$$

Mục tiêu:

$$
\boxed{
s(u,i)>s(u,j)
}
$$

Đây là **ranking objective**.

---

Một contrastive objective phổ biến như InfoNCE có dạng:

$$
\mathcal L_{CL}
=
-
\log
\frac{
\exp(sim(z,z^+)/\tau)
}{
\exp(sim(z,z^+)/\tau)
+
\sum_{z^-}
\exp(sim(z,z^-)/\tau)
}
$$

Mục tiêu:

$$
sim(z,z^+)\uparrow
$$

và:

$$
sim(z,z^-)\downarrow
$$

Contrastive loss không phải loss chính của paper LightGCN gốc. Các công trình recommendation sau này thường sử dụng LightGCN như encoder rồi thêm contrastive/self-supervised objectives.

Có thể nhớ:

$$
\boxed{
\text{BPR: positive item phải rank cao hơn negative item}
}
$$

$$
\boxed{
\text{Contrastive: positive representations phải gần hơn negatives}
}
$$

---

# 24. Toàn bộ LightGCN trong một chuỗi công thức

## Input graph

$$
A=
\begin{bmatrix}
0&R\\
R^T&0
\end{bmatrix}
$$

## Normalize

$$
\tilde A
=
D^{-1/2}AD^{-1/2}
$$

## Initial trainable embeddings

$$
E^{(0)}
$$

## Propagation

$$
E^{(k+1)}
=
\tilde AE^{(k)}
$$

hay:

$$
E^{(k)}
=
\tilde A^kE^{(0)}
$$

## Layer combination

$$
E
=
\sum_{k=0}^{K}
\alpha_kE^{(k)}
$$

## Prediction

$$
\hat y_{ui}
=
e_u^Te_i
$$

## Ranking loss

$$
\mathcal L_{BPR}
=
-\log
\sigma(
\hat y_{ui}
-
\hat y_{uj}
)
$$

## Final objective

$$
\boxed{
\mathcal L
=
\mathcal L_{BPR}
+
\lambda
\|E^{(0)}\|^2
}
$$

---

# 25. Công thức quan trọng nhất cần nhớ

Nếu chỉ cần nhớ 5 công thức của LightGCN:

### 1. Normalized adjacency

$$
\boxed{
\tilde A
=
D^{-1/2}AD^{-1/2}
}
$$

### 2. Graph propagation

$$
\boxed{
E^{(k+1)}
=
\tilde AE^{(k)}
}
$$

### 3. Multi-layer embedding

$$
\boxed{
E
=
\sum_{k=0}^{K}
\alpha_kE^{(k)}
}
$$

### 4. Prediction

$$
\boxed{
\hat y_{ui}
=
e_u^Te_i
}
$$

### 5. BPR loss

$$
\boxed{
\mathcal L
=
-
\log\sigma(
\hat y_{ui}-\hat y_{uj}
)
+
\lambda\|E^{(0)}\|^2
}
$$

---

# 26. Mental model

Một cách hiểu ngắn gọn:

```text
             User–Item Graph
                   │
                   ▼
          Initial Embeddings E⁽⁰⁾
                   │
                   ▼
        normalized graph diffusion
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
       E⁽¹⁾       E⁽²⁾       E⁽³⁾
        └──────────┼──────────┘
                   │
            weighted average
                   │
                   ▼
           Final Embeddings
              /          \
             /            \
          User u          Item i
             \            /
              \          /
                dot product
                   │
                   ▼
                score
                   │
                   ▼
               BPR Loss
                   │
                   ▼
            update E⁽⁰⁾
```

LightGCN không cố học một neural transformation phức tạp trên graph.

Nó để **graph structure quyết định cách information được truyền**, còn trainable embeddings quyết định latent preference của users/items.

---

# 27. Takeaways

1. **Initial embeddings là trainable dense ID embeddings**, không phải one-hot vectors được propagate trực tiếp.

2. LightGCN bỏ feature transformation và nonlinear activation của GCN.

3. Core operation:

$$
E^{(k+1)}
=
\tilde AE^{(k)}
$$

4. Higher layers capture high-order collaborative signals.

5. Final representation sử dụng weighted sum của nhiều propagation depths.

6. Prediction vẫn đơn giản là inner product như Matrix Factorization.

7. BPR trực tiếp optimize pairwise ranking:

$$
positive > negative
$$

8. Trainable parameters chủ yếu nằm ở:

$$
E^{(0)}
$$

9. Có thể xem LightGCN như:

$$
\boxed{
\text{Matrix Factorization + Linear Graph Diffusion}
}
$$

---

## Tài liệu tham khảo

- Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang. **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.** SIGIR 2020.
- arXiv: https://arxiv.org/abs/2002.02126
