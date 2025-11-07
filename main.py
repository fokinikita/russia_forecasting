from pipelines.run_gb_test import run_main_gb
from pipelines.run_ngb_test import run_main_ngb
from pipelines.run_tabnet_test import run_main_tabnet
from pipelines.run_metrics import run_metrics

if __name__ == '__main__':
    run_main_gb()
    run_main_ngb()
    run_main_tabnet()
    run_metrics()