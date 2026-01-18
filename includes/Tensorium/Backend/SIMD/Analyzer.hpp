#pragma once

#if defined(__GNUC__) || defined(__clang__)
#    define TENSORIUM_DUMP __attribute__((noinline, used, annotate("tensorium_dump")))
#    define TENSORIUM_ANNOTATE __attribute__((annotate("tensorium_dump")))

#else
#    define TENSORIUM_DUMP
#    define TENSORIUM_ANNOTATE
#endif
