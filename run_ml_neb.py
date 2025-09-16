# run_mlneb_qe_full.py

import sys
import copy
from ase.io import read
from ase.optimize import BFGS
from ase.calculators.espresso import Espresso
from catlearn.optimize.mlneb import MLNEB

# ===========================================================================
# Part 1: 전역 설정 (사용자가 수정할 부분)
# ===========================================================================

# --- 구조 최적화 설정 ---
# 최적화할 초기/최종 구조 파일
UNOPTIMIZED_INITIAL = 'initial.traj'
UNOPTIMIZED_FINAL = 'final.traj'
# 구조 최적화 수렴 기준 (힘, eV/Å 단위)
OPTIMIZE_FMAX = 0.1

# --- NEB 계산 설정 ---
# 최적화 후 생성될 구조 파일 (ML-NEB의 입력으로 사용됨)
OPTIMIZED_INITIAL = 'initial.traj'
OPTIMIZED_FINAL = 'final.traj'
# NEB 계산에 사용할 이미지 개수 (양 끝점 제외)
N_IMAGES = 5
# NEB 계산 수렴 기준 (힘, eV/Å 단위)
NEB_FMAX = 0.1
# 계산 결과 궤적 파일명
TRAJECTORY_FILE = 'mlneb_final.traj'


# ===========================================================================
# Part 2: Quantum Espresso 계산기 설정
# ===========================================================================
pseudopotentials = {
    'La': 'La.pbe-nsp-van.UPF',
    'Co': 'Co.pbe-nd-rrkjus.UPF',
    'O': 'O.pbe-rrkjus.UPF',
    'C': 'C.pbe-rrkjus.UPF',
    'H': 'H.pbe-rrkjus.UPF'
}
qe_input_data = {
    'control': {'calculation':'scf',
                'prefix':'qe_calc',
                'pseudo_dir': '/root/pseudo/',
                'outdir':'./out/',
                'tstress':True,
                'tprnfor':True},
    'system': {'ecutwfc':25,
               'ecutrho':225,
               'occupations':'smearing',
               'smearing':'gaussian',
               'degauss':0.01,
               'nspin':1},
    'electrons': {'conv_thr': 1.0e-4,
                  'mixing_beta': 0.1},
}
ase_calculator = Espresso(
    label='qe_calc',
    pseudopotentials=pseudopotentials,
    input_data=qe_input_data,
    kpts=(2, 1, 1),
    command='mpirun -np 12 pw.x -in PREFIX.pwi > PREFIX.pwo'
)

# ===========================================================================
# Part 3: 초기 & 최종 구조 최적화 (새로 추가된 부분)
# ===========================================================================
def optimize_endpoints():
    """초기/최종 구조를 BFGS 알고리즘으로 최적화하는 함수"""
    print("\n" + "="*60)
    print(">>> Part 3: 초기/최종 구조 최적화를 시작합니다.")
    
    # --- 초기 구조 최적화 ---
    print("\n>>> 1. Initial 구조 최적화를 진행합니다...")
    initial_atoms = read(UNOPTIMIZED_INITIAL)
    initial_atoms.set_calculator(copy.deepcopy(ase_calculator))
    optimizer_initial = BFGS(initial_atoms, trajectory=OPTIMIZED_INITIAL)
    optimizer_initial.run(fmax=OPTIMIZE_FMAX)
    print(f">>> Initial 구조 최적화 완료! 결과가 '{OPTIMIZED_INITIAL}'에 저장되었습니다.")

    # --- 최종 구조 최적화 ---
    print("\n>>> 2. Final 구조 최적화를 진행합니다...")
    final_atoms = read(UNOPTIMIZED_FINAL)
    final_atoms.set_calculator(copy.deepcopy(ase_calculator))
    optimizer_final = BFGS(final_atoms, trajectory=OPTIMIZED_FINAL)
    optimizer_final.run(fmax=OPTIMIZE_FMAX)
    print(f">>> Final 구조 최적화 완료! 결과가 '{OPTIMIZED_FINAL}'에 저장되었습니다.")
    print("="*60)


# ===========================================================================
# Part 4: ML-NEB 실행
# ===========================================================================
def run_main_mlneb():
    """메인 ML-NEB 계산을 실행하는 함수"""
    print("\n" + "="*60)
    print(">>> Part 4: ML-NEB 계산을 시작합니다.")
    
    mlneb = MLNEB(start=OPTIMIZED_INITIAL,
                  end=OPTIMIZED_FINAL,
                  ase_calc=copy.deepcopy(ase_calculator),
                  n_images=N_IMAGES,
                  k=0.1)
    # fmax는 run 메서드에 직접 전달합니다.
    mlneb.run(fmax=NEB_FMAX, trajectory=TRAJECTORY_FILE)
    print("\n>>> ML-NEB 계산이 성공적으로 완료되었습니다!")

# ===========================================================================
# Part 5: (선택) 단일 이미지 테스트 함수
# ===========================================================================
def run_single_point_test():
    """단일 이미지 SCF 테스트 함수 (최적화되지 않은 초기 구조로 테스트)"""
    print("\n>>> 단일 이미지 SCF 테스트를 시작합니다 (대상: UNOPTIMIZED_INITIAL)...")
    try:
        image = read(UNOPTIMIZED_INITIAL)
        image.set_calculator(copy.deepcopy(ase_calculator))
        energy = image.get_potential_energy()
        print(f">>> 테스트 성공! 에너지: {energy:.4f} eV")
    except Exception as e:
        print(f"!!! 테스트 실패: {e}!!!")
        print(">>> Part 2의 QE 계산기 설정을 다시 확인해 주세요.")

# ===========================================================================
# 스크립트 실행 제어
# ===========================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'test':
        run_single_point_test()
    else:
        # 1. 구조 최적화 실행
        #optimize_endpoints()
        # 2. ML-NEB 계산 실행
        run_main_mlneb()
