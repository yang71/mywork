#!/bin/sh
cd /home/yjy/gamma/gammagl/examples/gmm

for backend in 'tensorflow'
    do for n_kernels in 3 4 5
        do for lr in 0.01 0.005 0.001
            do for l2_coef in 1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3
                do for drop_rate in 0.7 0.75 0.8 0.9
                    do for hidden_dim in 8 16 32
                        do for i in 1 2 3 4 5
                            do
                                python3 gmm_trainer.py --n_kernels ${n_kernels} --lr ${lr} --l2_coef ${l2_coef} --drop_rate ${drop_rate} --hidden_dim ${hidden_dim}
                            done
                        done
                    done
                done
            done
        done
    done
done
