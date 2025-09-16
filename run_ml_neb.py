# run_mlneb_direct.py

import sys
import copy
from ase.io import read
from ase.calculators.espresso import Espresso
from catlearn.optimize.mlneb import MLNEB
import traceback

# ===========================================================================
# Part 1: 전역 설정 (사용자가 수정할 부분)
# ===========================================================================

# --- 입력 파일 ---
# ML-NEB 계산에 사용할 초기/최종 구조 파일
INITIAL_FILE = 'initial.traj'
FINAL_FILE = 'final.traj'

# --- NEB 계산 설정 ---
# NEB 계산에 사용할 이미지 개수 (양 끝점 제외)
N_IMAGES = 5
# NEB 계산 수렴 기준 (힘, eV/Å 단위)
NEB_FMAX = 0.1
# 계산 결과 궤적 파일명
TRAJECTORY_FILE = 'mlneb_final.traj'

# --- 계산 리소스 ---
N_CORES = 12

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
                'pseudo_dir': '/root/MLNEB/pseudo/',
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

# --- 최신 ASE 문법 적용 ---
from ase.calculators.espresso import EspressoProfile
command = f'mpirun --allow-run-as-root -x ESPRESSO_PSEUDO -np {N_CORES} pw.x'
profile = EspressoProfile(command=command, pseudo_dir=qe_input_data['control']['pseudo_dir'])

ase_calculator = Espresso(
    label='qe_calc',
    pseudopotentials=pseudopotentials,
    input_data=qe_input_data,
    kpts=(2, 1, 1),
    profile=profile
)

# ===========================================================================
# Part 3: ML-NEB 실행
# ===========================================================================
def run_main_mlneb():
    """메인 ML-NEB 계산을 실행하는 함수"""
    print("\n" + "="*60)
    print(">>> ML-NEB 계산을 시작합니다.")
    
    try:
        mlneb = MLNEB(start=INITIAL_FILE,
                      end=FINAL_FILE,
                      ase_calc=copy.deepcopy(ase_calculator),
                      n_images=N_IMAGES,
                      k=0.1)
        
        mlneb.run(fmax=NEB_FMAX, trajectory=TRAJECTORY_FILE)
        print("\n>>> ML-NEB 계산이 성공적으로 완료되었습니다!")
        
    except Exception:
        print("\n" + "="*60)
        print("!!! ML-NEB 계산 중 오류가 발생했습니다 !!!")
        print("--- 전체 오류 메시지 (Traceback) ---")
        traceback.print_exc()
        print("------------------------------------")
        print("\n>>> 'qe_calc.pwo' 또는 'out/' 폴더의 출력 파일을 확인하여 원인을 파악하세요.")
        print("="*60 + "\n")

# ===========================================================================
# 스크립트 실행 제어
# ===========================================================================
if __name__ == "__main__":
    run_main_mlneb()

