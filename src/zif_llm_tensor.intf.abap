"! <p class="shorttext synchronized" lang="en">LLM Engine - Tensor Interface</p>
"! Abstraction for multi-dimensional tensor operations used in
"! neural network inference. All operations return new tensor
"! instances (immutable pattern) unless otherwise noted.
INTERFACE zif_llm_tensor
  PUBLIC.

  TYPES:
    "! Table of integers representing tensor dimensions
    ty_shape    TYPE STANDARD TABLE OF i WITH EMPTY KEY,
    "! Table of float values for tensor data storage
    ty_float_tab TYPE STANDARD TABLE OF f WITH EMPTY KEY.

  METHODS:
    "! <p class="shorttext synchronized">Get tensor shape</p>
    "! @parameter rt_shape | Dimension sizes as integer table
    get_shape
      RETURNING VALUE(rt_shape) TYPE ty_shape,

    "! <p class="shorttext synchronized">Get total number of elements</p>
    "! @parameter rv_size | Total element count
    get_size
      RETURNING VALUE(rv_size) TYPE i,

    "! <p class="shorttext synchronized">Get raw float data</p>
    "! @parameter rt_data | Float table with tensor data
    get_data
      RETURNING VALUE(rt_data) TYPE ty_float_tab,

    "! <p class="shorttext synchronized">Direct access to internal data (for performance)</p>
    "! Use with care — breaks immutability.
    "! @parameter rr_data | Reference to internal float table
    get_data_ref
      RETURNING VALUE(rr_data) TYPE REF TO ty_float_tab,

    "! <p class="shorttext synchronized">Get a single element by flat index</p>
    "! @parameter iv_index | 0-based flat index
    "! @parameter rv_value | Float value
    get_value
      IMPORTING iv_index        TYPE i
      RETURNING VALUE(rv_value) TYPE f,

    "! <p class="shorttext synchronized">Set a single element by flat index</p>
    "! @parameter iv_index | 0-based flat index
    "! @parameter iv_value | Float value
    set_value
      IMPORTING iv_index TYPE i
                iv_value TYPE f,

    "! <p class="shorttext synchronized">Matrix multiplication</p>
    "! Performs this @ other. For 2D tensors: (M,K) @ (K,N) = (M,N).
    "! Uses tiled algorithm for cache efficiency.
    "! @parameter io_other | Right-hand tensor
    "! @parameter ro_result | Result tensor
    matmul
      IMPORTING io_other        TYPE REF TO zif_llm_tensor
      RETURNING VALUE(ro_result) TYPE REF TO zif_llm_tensor
      RAISING   cx_sy_arithmetic_overflow,

    "! <p class="shorttext synchronized">Element-wise addition</p>
    "! @parameter io_other | Tensor to add
    "! @parameter ro_result | Result tensor
    add
      IMPORTING io_other        TYPE REF TO zif_llm_tensor
      RETURNING VALUE(ro_result) TYPE REF TO zif_llm_tensor,

    "! <p class="shorttext synchronized">Element-wise multiplication</p>
    "! @parameter io_other | Tensor to multiply element-wise
    "! @parameter ro_result | Result tensor
    multiply_elementwise
      IMPORTING io_other        TYPE REF TO zif_llm_tensor
      RETURNING VALUE(ro_result) TYPE REF TO zif_llm_tensor,

    "! <p class="shorttext synchronized">Scalar multiplication</p>
    "! @parameter iv_factor | Scalar multiplier
    "! @parameter ro_result | Result tensor
    scale
      IMPORTING iv_factor       TYPE f
      RETURNING VALUE(ro_result) TYPE REF TO zif_llm_tensor,

    "! <p class="shorttext synchronized">Reshape tensor</p>
    "! Total element count must remain the same.
    "! @parameter it_shape | New shape
    "! @parameter ro_result | Reshaped tensor (shares data)
    reshape
      IMPORTING it_shape        TYPE ty_shape
      RETURNING VALUE(ro_result) TYPE REF TO zif_llm_tensor,

    "! <p class="shorttext synchronized">Slice contiguous elements</p>
    "! Returns a 1D view starting at iv_start with iv_length elements.
    "! @parameter iv_start | Start index (0-based)
    "! @parameter iv_length | Number of elements
    "! @parameter ro_result | Sliced tensor
    slice
      IMPORTING iv_start        TYPE i
                iv_length       TYPE i
      RETURNING VALUE(ro_result) TYPE REF TO zif_llm_tensor.

ENDINTERFACE.
