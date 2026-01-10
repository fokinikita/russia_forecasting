from pipelines.run_dfm_test import run_main_dfm
from pipelines.run_gb_test import run_main_gb
from pipelines.run_metrics import run_metrics
from pipelines.run_ngb_test import run_main_ngb
from pipelines.run_tabnet_test import run_main_tabnet

if __name__ == "__main__":
    run_main_gb() # считается быстро
    run_main_ngb() # считается быстро
    run_main_tabnet() # считается быстро
    run_main_dfm() # считается долго, много опций гиперпараметров, можно не запускать без необходимости (закоментить)
    run_metrics(run_mfbvar=True)
