"""
iPUF (Interpose PUF) 仿真模块

基于: "Splitting the Interpose PUF" (Wisiol et al., CHES 2020)

iPUF 结构:
    - 上层: k_up 个标准 Arbiter PUF，输出 r_up
    - 下层: k_down 个标准 Arbiter PUF
    - Interpose: r_up 被插入到下层挑战的特定位置（默认第 0 位）
    - 最终响应: r = r_up XOR r_down

安全声称: 等效于 (k_down + k_up/2)-XOR APUF
实际安全性 (拆分攻击): 等效于 max(k_up, k_down)-XOR APUF

参考:
    Wisiol, N., et al. "Splitting the Interpose PUF: A Novel Modeling Attack Strategy."
    IACR TCHES, Vol. 2020, Issue 3, pp. 97-120.
"""

import numpy as np


class IPUFModel:
    """
    Interpose PUF (iPUF) 仿真模型。

    结构:
        (k_up, k_down)-iPUF
        - Upper: k_up 个 APUF，延迟权重 w_up
        - Lower: k_down 个 APUF，延迟权重 w_down
        - Interpose: 上层响应替换下层挑战的 interpose_pos 位

    响应计算:
        1. r_up = XOR_i( sign(phi · w_up[i, :-1] + w_up[i, -1]) )
        2. phi' = phi.copy(); phi'[interpose_pos] = r_up
        3. r_down = XOR_i( sign(phi' · w_down[i, :-1] + w_down[i, -1]) )
        4. r = r_up XOR r_down
    """

    def __init__(
        self,
        k_up: int,
        k_down: int,
        length: int = 64,
        weight_up: np.ndarray | None = None,
        weight_down: np.ndarray | None = None,
        alpha: float = 0.05,
        noise: float = 0.0,
        interpose_pos: int = 0,
    ):
        """
        Args:
            k_up: 上层 XOR APUF 数量
            k_down: 下层 XOR APUF 数量
            length: 挑战位长度（默认 64）
            weight_up: 上层权重矩阵 [k_up, length+1]，None 则随机生成
            weight_down: 下层权重矩阵 [k_down, length+1]，None 则随机生成
            alpha: 权重标准差（生成时使用）
            noise: 噪声标准差（相对 alpha 的比例）
            interpose_pos: Interpose 位置（r_up 插入到下层挑战的位置，默认 0）
        """
        self.k_up = k_up
        self.k_down = k_down
        self.length = length
        self.noise = noise
        self.alpha = alpha
        self.interpose_pos = interpose_pos

        # 权重初始化
        if weight_up is not None:
            self.weight_up = weight_up
        else:
            self.weight_up = np.random.normal(0, alpha, size=(k_up, length + 1))

        if weight_down is not None:
            self.weight_down = weight_down
        else:
            self.weight_down = np.random.normal(0, alpha, size=(k_down, length + 1))

    def getResponse(self, phi: np.ndarray) -> int:
        """
        计算 iPUF 对单个挑战的响应。

        Args:
            phi: 挑战向量 [length] 或 [batch, length]，值 ∈ {0, 1}

        Returns:
            响应值 0 或 1
        """
        phi = np.asarray(phi)
        single = (phi.ndim == 1)
        if single:
            phi = phi.reshape(1, -1)

        batch_size = phi.shape[0]
        responses = np.zeros(batch_size, dtype=int)

        for b in range(batch_size):
            challenge = phi[b].copy()

            # ---- 上层: k_up 个 APUF 的 XOR ----
            # 注入噪声
            noise_up = np.random.normal(
                0, self.alpha * self.noise, size=self.weight_up.shape
            )
            w_up = self.weight_up + noise_up
            w_up_feat = w_up[:, :-1]  # [k_up, length]
            w_up_bias = w_up[:, -1]   # [k_up]

            delays_up = np.dot(w_up_feat, challenge) + w_up_bias  # [k_up]
            r_up = 0
            for i in range(self.k_up):
                r_up ^= int(delays_up[i] >= 0)

            # ---- Interpose: r_up 插入下层挑战 ----
            challenge_modified = challenge.copy()
            challenge_modified[self.interpose_pos] = r_up

            # ---- 下层: k_down 个 APUF 的 XOR ----
            noise_down = np.random.normal(
                0, self.alpha * self.noise, size=self.weight_down.shape
            )
            w_down = self.weight_down + noise_down
            w_down_feat = w_down[:, :-1]  # [k_down, length]
            w_down_bias = w_down[:, -1]   # [k_down]

            delays_down = np.dot(w_down_feat, challenge_modified) + w_down_bias
            r_down = 0
            for i in range(self.k_down):
                r_down ^= int(delays_down[i] >= 0)

            # ---- 最终响应 ----
            responses[b] = r_up ^ r_down

        return responses[0] if single else responses

    @classmethod
    def randomSample(
        cls,
        k_up: int = 2,
        k_down: int = 2,
        length: int = 64,
        alpha: float = 0.05,
        noise: float = 0.0,
        interpose_pos: int = 0,
    ) -> "IPUFModel":
        """
        生成随机 iPUF 实例。

        Args:
            k_up: 上层 XOR 数量
            k_down: 下层 XOR 数量
            length: 挑战位长度
            alpha: 权重标准差
            noise: 噪声水平
            interpose_pos: Interpose 位置

        Returns:
            随机初始化的 IPUFModel
        """
        return cls(
            k_up=k_up,
            k_down=k_down,
            length=length,
            alpha=alpha,
            noise=noise,
            interpose_pos=interpose_pos,
        )

    @property
    def effective_xor(self) -> int:
        """声称的等效 XOR 安全性: k_down + k_up/2"""
        return self.k_down + self.k_up // 2

    @property
    def actual_xor(self) -> int:
        """拆分攻击下的实际等效 XOR 安全性: max(k_up, k_down)"""
        return max(self.k_up, self.k_down)

    def save(self, filename: str):
        """保存权重到文件"""
        np.savez(
            filename,
            weight_up=self.weight_up,
            weight_down=self.weight_down,
            k_up=self.k_up,
            k_down=self.k_down,
            length=self.length,
            interpose_pos=self.interpose_pos,
        )

    @classmethod
    def load(cls, filename: str) -> "IPUFModel":
        """从文件加载权重"""
        data = np.load(filename)
        return cls(
            k_up=int(data["k_up"]),
            k_down=int(data["k_down"]),
            length=int(data["length"]),
            weight_up=data["weight_up"],
            weight_down=data["weight_down"],
            interpose_pos=int(data["interpose_pos"]),
        )

    def __repr__(self) -> str:
        return (
            f"IPUFModel(k_up={self.k_up}, k_down={self.k_down}, "
            f"length={self.length}, interpose_pos={self.interpose_pos}, "
            f"effective_xor={self.effective_xor}, "
            f"actual_xor={self.actual_xor})"
        )
