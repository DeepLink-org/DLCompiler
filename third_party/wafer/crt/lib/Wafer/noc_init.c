// Initialize the protocol-owned words before a 16-tile NoC launch.
#include <stdint.h>
#include "tx81_spm.h"
#include "wafer.h"

void __NoCRingInit(void) {
  volatile uint32_t *state =
      (volatile uint32_t *)get_spm_memory_mapping(SINGLE_SPM_SYNC_ADDR);
  state[0] = 0; // request
  state[1] = 0; // acknowledgement
#ifdef __riscv
  __asm__ __volatile__("fence iorw, iorw" ::: "memory");
#else
  __sync_synchronize();
#endif
}
