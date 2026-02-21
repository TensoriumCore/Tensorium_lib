#pragma once

#ifdef _OPENMP
extern "C" {
int omp_get_max_threads(void);
int omp_get_thread_num(void);
}
#endif
