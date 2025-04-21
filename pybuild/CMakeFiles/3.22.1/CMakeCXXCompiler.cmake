set(CMAKE_CXX_COMPILER "/nix/store/mpm3i0sbqc9svfch6a17179fs64dz2kv-gcc-wrapper-13.3.0/bin/g++")
set(CMAKE_CXX_COMPILER_ARG1 "")
set(CMAKE_CXX_COMPILER_ID "GNU")
set(CMAKE_CXX_COMPILER_VERSION "13.3.0")
set(CMAKE_CXX_COMPILER_VERSION_INTERNAL "")
set(CMAKE_CXX_COMPILER_WRAPPER "")
set(CMAKE_CXX_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CXX_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CXX_COMPILE_FEATURES "cxx_std_98;cxx_template_template_parameters;cxx_std_11;cxx_alias_templates;cxx_alignas;cxx_alignof;cxx_attributes;cxx_auto_type;cxx_constexpr;cxx_decltype;cxx_decltype_incomplete_return_types;cxx_default_function_template_args;cxx_defaulted_functions;cxx_defaulted_move_initializers;cxx_delegating_constructors;cxx_deleted_functions;cxx_enum_forward_declarations;cxx_explicit_conversions;cxx_extended_friend_declarations;cxx_extern_templates;cxx_final;cxx_func_identifier;cxx_generalized_initializers;cxx_inheriting_constructors;cxx_inline_namespaces;cxx_lambdas;cxx_local_type_template_args;cxx_long_long_type;cxx_noexcept;cxx_nonstatic_member_init;cxx_nullptr;cxx_override;cxx_range_for;cxx_raw_string_literals;cxx_reference_qualified_functions;cxx_right_angle_brackets;cxx_rvalue_references;cxx_sizeof_member;cxx_static_assert;cxx_strong_enums;cxx_thread_local;cxx_trailing_return_types;cxx_unicode_literals;cxx_uniform_initialization;cxx_unrestricted_unions;cxx_user_literals;cxx_variadic_macros;cxx_variadic_templates;cxx_std_14;cxx_aggregate_default_initializers;cxx_attribute_deprecated;cxx_binary_literals;cxx_contextual_conversions;cxx_decltype_auto;cxx_digit_separators;cxx_generic_lambdas;cxx_lambda_init_captures;cxx_relaxed_constexpr;cxx_return_type_deduction;cxx_variable_templates;cxx_std_17;cxx_std_20;cxx_std_23")
set(CMAKE_CXX98_COMPILE_FEATURES "cxx_std_98;cxx_template_template_parameters")
set(CMAKE_CXX11_COMPILE_FEATURES "cxx_std_11;cxx_alias_templates;cxx_alignas;cxx_alignof;cxx_attributes;cxx_auto_type;cxx_constexpr;cxx_decltype;cxx_decltype_incomplete_return_types;cxx_default_function_template_args;cxx_defaulted_functions;cxx_defaulted_move_initializers;cxx_delegating_constructors;cxx_deleted_functions;cxx_enum_forward_declarations;cxx_explicit_conversions;cxx_extended_friend_declarations;cxx_extern_templates;cxx_final;cxx_func_identifier;cxx_generalized_initializers;cxx_inheriting_constructors;cxx_inline_namespaces;cxx_lambdas;cxx_local_type_template_args;cxx_long_long_type;cxx_noexcept;cxx_nonstatic_member_init;cxx_nullptr;cxx_override;cxx_range_for;cxx_raw_string_literals;cxx_reference_qualified_functions;cxx_right_angle_brackets;cxx_rvalue_references;cxx_sizeof_member;cxx_static_assert;cxx_strong_enums;cxx_thread_local;cxx_trailing_return_types;cxx_unicode_literals;cxx_uniform_initialization;cxx_unrestricted_unions;cxx_user_literals;cxx_variadic_macros;cxx_variadic_templates")
set(CMAKE_CXX14_COMPILE_FEATURES "cxx_std_14;cxx_aggregate_default_initializers;cxx_attribute_deprecated;cxx_binary_literals;cxx_contextual_conversions;cxx_decltype_auto;cxx_digit_separators;cxx_generic_lambdas;cxx_lambda_init_captures;cxx_relaxed_constexpr;cxx_return_type_deduction;cxx_variable_templates")
set(CMAKE_CXX17_COMPILE_FEATURES "cxx_std_17")
set(CMAKE_CXX20_COMPILE_FEATURES "cxx_std_20")
set(CMAKE_CXX23_COMPILE_FEATURES "cxx_std_23")

set(CMAKE_CXX_PLATFORM_ID "Linux")
set(CMAKE_CXX_SIMULATE_ID "")
set(CMAKE_CXX_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CXX_SIMULATE_VERSION "")




set(CMAKE_AR "/nix/store/mpm3i0sbqc9svfch6a17179fs64dz2kv-gcc-wrapper-13.3.0/bin/ar")
set(CMAKE_CXX_COMPILER_AR "/nix/store/zc0nsv23pakbafngjy32kvhfzb16as43-gcc-13.3.0/bin/gcc-ar")
set(CMAKE_RANLIB "/nix/store/mpm3i0sbqc9svfch6a17179fs64dz2kv-gcc-wrapper-13.3.0/bin/ranlib")
set(CMAKE_CXX_COMPILER_RANLIB "/nix/store/zc0nsv23pakbafngjy32kvhfzb16as43-gcc-13.3.0/bin/gcc-ranlib")
set(CMAKE_LINKER "/nix/store/mpm3i0sbqc9svfch6a17179fs64dz2kv-gcc-wrapper-13.3.0/bin/ld")
set(CMAKE_MT "")
set(CMAKE_COMPILER_IS_GNUCXX 1)
set(CMAKE_CXX_COMPILER_LOADED 1)
set(CMAKE_CXX_COMPILER_WORKS TRUE)
set(CMAKE_CXX_ABI_COMPILED TRUE)

set(CMAKE_CXX_COMPILER_ENV_VAR "CXX")

set(CMAKE_CXX_COMPILER_ID_RUN 1)
set(CMAKE_CXX_SOURCE_FILE_EXTENSIONS C;M;c++;cc;cpp;cxx;m;mm;mpp;CPP;ixx;cppm)
set(CMAKE_CXX_IGNORE_EXTENSIONS inl;h;hpp;HPP;H;o;O;obj;OBJ;def;DEF;rc;RC)

foreach (lang C OBJC OBJCXX)
  if (CMAKE_${lang}_COMPILER_ID_RUN)
    foreach(extension IN LISTS CMAKE_${lang}_SOURCE_FILE_EXTENSIONS)
      list(REMOVE_ITEM CMAKE_CXX_SOURCE_FILE_EXTENSIONS ${extension})
    endforeach()
  endif()
endforeach()

set(CMAKE_CXX_LINKER_PREFERENCE 30)
set(CMAKE_CXX_LINKER_PREFERENCE_PROPAGATES 1)

# Save compiler ABI information.
set(CMAKE_CXX_SIZEOF_DATA_PTR "8")
set(CMAKE_CXX_COMPILER_ABI "ELF")
set(CMAKE_CXX_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CXX_LIBRARY_ARCHITECTURE "")

if(CMAKE_CXX_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CXX_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CXX_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CXX_COMPILER_ABI}")
endif()

if(CMAKE_CXX_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_CXX_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "/nfs/homes/ltouzali/.local/include/hdf5/include;/nix/store/zmk51gggsp568b9idaj6dr17wwb2y8rv-numactl-2.0.18-dev/include;/nix/store/dq03m14rqjvbz6pw42g555df4khh4bvr-zlib-1.3.1-dev/include;/nix/store/x9avi0hqjd27nrgxmhhlnp1ygybyhss5-systemd-minimal-libs-255.6-dev/include;/nix/store/wkzw5axfagxfxcc3pcshpg7qk1ml6458-openmpi-4.1.6-dev/include;/nix/store/fiffklfh51i0rm0g3smnlnw6idbc7wh0-openmp-17.0.6-dev/include;/nix/store/xbacjhalixj0hynmpfib71qpj3w6lc76-fftw-double-3.3.10-dev/include;/nix/store/qjl1wmjh7y8mkfy4kprg9dfpi2f283cf-eigen-3.4.0/include;/nix/store/bdvmya35kgwgk4x6ifz639jrjmrk6di3-openblas-0.3.27-dev/include;/nix/store/sfr7kcb8b5z39fjjyrvfgmamd7ffv440-python3.12-pybind11-2.12.0/include;/nix/store/zwvdwiw0zg5liz6lkkkqyab035parhld-python3-3.12.3/include;/nix/store/nw9rz88ws6hjh3fiavzjg5ljfsgwc9cx-lapack-3-dev/include;/nix/store/vx1p7fsbgkdylg8k3k28p8zyd8z2pfk0-compiler-rt-libc-17.0.6-dev/include;/nix/store/zc0nsv23pakbafngjy32kvhfzb16as43-gcc-13.3.0/include/c++/13.3.0;/nix/store/zc0nsv23pakbafngjy32kvhfzb16as43-gcc-13.3.0/include/c++/13.3.0/x86_64-unknown-linux-gnu;/nix/store/zc0nsv23pakbafngjy32kvhfzb16as43-gcc-13.3.0/include/c++/13.3.0/backward;/nix/store/zc0nsv23pakbafngjy32kvhfzb16as43-gcc-13.3.0/lib/gcc/x86_64-unknown-linux-gnu/13.3.0/include;/nix/store/zc0nsv23pakbafngjy32kvhfzb16as43-gcc-13.3.0/include;/nix/store/zc0nsv23pakbafngjy32kvhfzb16as43-gcc-13.3.0/lib/gcc/x86_64-unknown-linux-gnu/13.3.0/include-fixed;/nix/store/4vgk1rlzdqjnpjicb5qcxjcd4spi7wyw-glibc-2.39-52-dev/include")
set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "/nix/store/ywhx81lh73qwfj86fldfarz5q5rnwihg-numactl-2.0.18/lib;/nix/store/971g0fka30djkcww2ddpx7q66dr86c1q-zlib-1.3.1/lib;/nix/store/ncpzg5dklh1qvhj8lrdwx36hwlxl0ry1-systemd-minimal-libs-255.6/lib;/nix/store/21xkx8bp4biq81dhpqplkwxfw7fj91x0-openmpi-4.1.6/lib;/nix/store/jvxkn1lcqid261g7gmimn4p5bf6zjnhk-openmp-17.0.6/lib;/nix/store/pm7q9g2zxqk0n4xln21j0nbii1h2a7qj-fftw-double-3.3.10/lib;/nix/store/c2f2cvh74ynpf06ymbkj85c2405w99qz-openblas-0.3.27/lib;/nix/store/zwvdwiw0zg5liz6lkkkqyab035parhld-python3-3.12.3/lib;/nix/store/mc3hg91g97sryr7zpwaw2gw3n5vl72vr-lapack-3/lib;/nix/store/dbwp0scbb0rk78m636sb7cvycz8xzgyh-glibc-2.39-52/lib;/nix/store/zc0nsv23pakbafngjy32kvhfzb16as43-gcc-13.3.0/lib/gcc/x86_64-unknown-linux-gnu/13.3.0;/nix/store/bn7pnigb0f8874m6riiw6dngsmdyic1g-gcc-13.3.0-lib/lib;/nix/store/mpm3i0sbqc9svfch6a17179fs64dz2kv-gcc-wrapper-13.3.0/bin;/nix/store/zc0nsv23pakbafngjy32kvhfzb16as43-gcc-13.3.0/lib64;/nix/store/zc0nsv23pakbafngjy32kvhfzb16as43-gcc-13.3.0/lib")
set(CMAKE_CXX_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
