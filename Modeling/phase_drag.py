# phase_drag.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import networkx as nx
from networkx.algorithms import bipartite

class QaetherSimulation:
    def __init__(self, coupling_g=1.0, dt=0.1):
        self.n_faces = 8
        self.coupling_g = coupling_g
        self.dt = dt
        
        # 1. 정팔면체 면 그래프 구축 (정육면체 그래프)
        self.build_octahedron_graph()
        
        # 2. 색 구획 정의 (수동)
        self.define_color_sectors()
        
        # 3. 초기 상태 설정
        self.reset_state()
        
        # 4. 시뮬레이션 히스토리
        self.history = []
        
    def build_octahedron_graph(self):
        """정팔면체의 면 그래프(정육면체 그래프) 구축"""
        # 3차원 하이퍼큐브(정육면체) 그래프: 8개 노드
        self.G = nx.hypercube_graph(3)
        self.adj = nx.to_numpy_array(self.G)
        # 시각화 레이아웃
        self.pos = nx.spring_layout(self.G, seed=42)
        
    def define_color_sectors(self):
        """8개 면을 가상의 색 구획으로 할당 (인접성 고려)"""
        # 실제 물리적 색과는 무관, 단지 시뮬레이션용 구분
        self.sector_R = [0, 4]  # f1, f5
        self.sector_G = [1, 5]  # f2, f6
        self.sector_B = [2, 3, 6, 7]  # 나머지
        
    def reset_state(self, state_type='R_active'):
        """초기 상태 재설정"""
        self.psi = np.zeros(self.n_faces, dtype=complex)
        
        if state_type == 'R_active':
            # R 섹터에만 에너지 집중 (정규화)
            self.psi[self.sector_R] = 1.0 / np.sqrt(len(self.sector_R))
        elif state_type == 'G_active':
            self.psi[self.sector_G] = 1.0 / np.sqrt(len(self.sector_G))
        elif state_type == 'B_active':
            self.psi[self.sector_B] = 1.0 / np.sqrt(len(self.sector_B))
        elif state_type == 'random':
            self.psi = np.random.randn(self.n_faces) + 1j * np.random.randn(self.n_faces)
            self.psi /= np.linalg.norm(self.psi)
        elif state_type == 'superposition':
            # R과 G의 중첩
            self.psi[self.sector_R] = 1.0 / np.sqrt(2 * len(self.sector_R))
            self.psi[self.sector_G] = 1.0 / np.sqrt(2 * len(self.sector_G))
            
    def construct_generator_hamiltonian(self, gen_type='lambda1', coupling=None):
        """
        SU(3) 생성자에 대응하는 해밀토니안 구성
        gen_type: 'lambda1' ~ 'lambda8'
        """
        if coupling is None:
            coupling = self.coupling_g
            
        H = np.zeros((self.n_faces, self.n_faces), dtype=complex)
        
        if gen_type == 'lambda1':  # R <-> G
            for r in self.sector_R:
                for g in self.sector_G:
                    if self.adj[r][g] == 1:
                        H[r][g] = 1.0
                        H[g][r] = 1.0
                        
        elif gen_type == 'lambda2':  # R <-> G (위상 90도)
            for r in self.sector_R:
                for g in self.sector_G:
                    if self.adj[r][g] == 1:
                        H[r][g] = -1j
                        H[g][r] = 1j
                        
        elif gen_type == 'lambda4':  # R <-> B
            for r in self.sector_R:
                for b in self.sector_B:
                    if self.adj[r][b] == 1:
                        H[r][b] = 1.0
                        H[b][r] = 1.0
                        
        elif gen_type == 'lambda5':  # R <-> B (위상 90도)
            for r in self.sector_R:
                for b in self.sector_B:
                    if self.adj[r][b] == 1:
                        H[r][b] = -1j
                        H[b][r] = 1j
                        
        elif gen_type == 'lambda6':  # G <-> B
            for g in self.sector_G:
                for b in self.sector_B:
                    if self.adj[g][b] == 1:
                        H[g][b] = 1.0
                        H[b][g] = 1.0
                        
        elif gen_type == 'lambda7':  # G <-> B (위상 90도)
            for g in self.sector_G:
                for b in self.sector_B:
                    if self.adj[g][b] == 1:
                        H[g][b] = -1j
                        H[b][g] = 1j
                        
        elif gen_type == 'lambda3':  # Isospin (대각)
            for i in range(self.n_faces):
                if i in self.sector_R:
                    H[i][i] = 1.0
                elif i in self.sector_G:
                    H[i][i] = -1.0
                # B 섹터는 0
                    
        elif gen_type == 'lambda8':  # Hypercharge (대각)
            for i in range(self.n_faces):
                if i in self.sector_R or i in self.sector_G:
                    H[i][i] = 1.0
                else:
                    H[i][i] = -2.0
            # 대각합 0 보정 (이미 0이지만 스케일 조정)
            H = H / np.sqrt(3)
            
        return coupling * H
    
    def run_step(self, gen_types=['lambda1'], couplings=None, dt=None):
        """여러 생성자 동시 작용 시간 발전"""
        if dt is None:
            dt = self.dt
        if couplings is None:
            couplings = [self.coupling_g] * len(gen_types)
            
        H_total = np.zeros((self.n_faces, self.n_faces), dtype=complex)
        for gt, c in zip(gen_types, couplings):
            H_total += self.construct_generator_hamiltonian(gt, c)
            
        # 유니타리 연산자
        U = expm(-1j * H_total * dt)
        self.psi = np.dot(U, self.psi)
        
        # 수치 오차 보정
        norm = np.linalg.norm(self.psi)
        if abs(norm - 1.0) > 1e-12:
            self.psi /= norm
            
        return np.abs(self.psi)**2
    
    def run_simulation(self, n_steps=100, gen_types=['lambda1'], couplings=None, record_every=1):
        """전체 시뮬레이션 실행"""
        self.history = []
        for step in range(n_steps):
            prob = self.run_step(gen_types, couplings)
            if step % record_every == 0:
                self.history.append({
                    'step': step,
                    'prob': prob.copy(),
                    'sector_R': np.sum(prob[self.sector_R]),
                    'sector_G': np.sum(prob[self.sector_G]),
                    'sector_B': np.sum(prob[self.sector_B]),
                })
        return self.history
    
    def plot_dynamics(self):
        """시간에 따른 섹터 확률 그래프"""
        if not self.history:
            print("No history to plot.")
            return
        
        steps = [h['step'] for h in self.history]
        R = [h['sector_R'] for h in self.history]
        G = [h['sector_G'] for h in self.history]
        B = [h['sector_B'] for h in self.history]
        
        plt.figure(figsize=(10, 5))
        plt.plot(steps, R, 'r-', label='Red sector', linewidth=2)
        plt.plot(steps, G, 'g-', label='Green sector', linewidth=2)
        plt.plot(steps, B, 'b-', label='Blue sector', linewidth=2)
        plt.xlabel('Time step')
        plt.ylabel('Probability')
        plt.title('Color sector dynamics')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

# ========== 실행 예제 ==========
if __name__ == "__main__":
    print("=== Qaether 시뮬레이션 시작 ===")
    
    # 1. λ1만 작용: R ↔ G 진동
    sim = QaetherSimulation(coupling_g=1.0, dt=0.2)
    sim.reset_state('R_active')
    sim.run_simulation(n_steps=50, gen_types=['lambda1'])
    print("λ1 시뮬레이션 완료, 마지막 상태:")
    last = sim.history[-1]
    print(f"  R: {last['sector_R']:.3f}, G: {last['sector_G']:.3f}, B: {last['sector_B']:.3f}")
    sim.plot_dynamics()
    
    # 2. λ1 + λ4: R이 G와 B로 퍼짐
    sim2 = QaetherSimulation(coupling_g=1.0, dt=0.2)
    sim2.reset_state('R_active')
    sim2.run_simulation(n_steps=50, gen_types=['lambda1', 'lambda4'])
    print("\nλ1+λ4 시뮬레이션 완료, 마지막 상태:")
    last = sim2.history[-1]
    print(f"  R: {last['sector_R']:.3f}, G: {last['sector_G']:.3f}, B: {last['sector_B']:.3f}")
    sim2.plot_dynamics()
    
    # 3. 모든 생성자 동시 작용 (완전 SU(3))
    sim3 = QaetherSimulation(coupling_g=1.0, dt=0.1)
    sim3.reset_state('R_active')
    all_8 = ['lambda1','lambda2','lambda3','lambda4','lambda5','lambda6','lambda7','lambda8']
    sim3.run_simulation(n_steps=80, gen_types=all_8)
    print("\n완전 SU(3) 시뮬레이션 완료, 마지막 상태:")
    last = sim3.history[-1]
    print(f"  R: {last['sector_R']:.3f}, G: {last['sector_G']:.3f}, B: {last['sector_B']:.3f}")
    sim3.plot_dynamics()
    
    print("\n=== 시뮬레이션 종료 ===")