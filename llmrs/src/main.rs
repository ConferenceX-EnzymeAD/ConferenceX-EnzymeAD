#![allow(dead_code, mutable_transmutes, non_camel_case_types, non_snake_case, non_upper_case_globals, unused_assignments, unused_mut)]
#![feature(autodiff,extern_types, linkage)]
use std::autodiff::autodiff;
extern "C" {
    pub type _IO_wide_data;
    pub type _IO_codecvt;
    pub type _IO_marker;
    pub type __dirstream;
    static mut stdout: *mut FILE;
    static mut stderr: *mut FILE;
    fn fclose(__stream: *mut FILE) -> libc::c_int;
    fn fflush(__stream: *mut FILE) -> libc::c_int;
    fn fopen(_: *const libc::c_char, _: *const libc::c_char) -> *mut FILE;
    fn fprintf(_: *mut FILE, _: *const libc::c_char, _: ...) -> libc::c_int;
    fn printf(_: *const libc::c_char, _: ...) -> libc::c_int;
    fn fread(
        _: *mut libc::c_void,
        _: libc::c_ulong,
        _: libc::c_ulong,
        _: *mut FILE,
    ) -> libc::c_ulong;
    fn fwrite(
        _: *const libc::c_void,
        _: libc::c_ulong,
        _: libc::c_ulong,
        _: *mut FILE,
    ) -> libc::c_ulong;
    fn fseek(
        __stream: *mut FILE,
        __off: libc::c_long,
        __whence: libc::c_int,
    ) -> libc::c_int;
    fn ftell(__stream: *mut FILE) -> libc::c_long;
    fn feof(__stream: *mut FILE) -> libc::c_int;
    fn ferror(__stream: *mut FILE) -> libc::c_int;
    fn atoi(__nptr: *const libc::c_char) -> libc::c_int;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn calloc(_: libc::c_ulong, _: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
    fn exit(_: libc::c_int) -> !;
    fn __ctype_b_loc() -> *mut *const libc::c_ushort;
    fn cosf(_: f32) -> f32;
    fn sinf(_: f32) -> f32;
    fn tanhf(_: f32) -> f32;
    fn expf(_: f32) -> f32;
    fn logf(_: f32) -> f32;
    fn powf(_: f32, _: f32) -> f32;
    fn sqrtf(_: f32) -> f32;
    fn clock_gettime(__clock_id: clockid_t, __tp: *mut timespec) -> libc::c_int;
    fn memset(
        _: *mut libc::c_void,
        _: libc::c_int,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
    fn strncmp(
        _: *const libc::c_char,
        _: *const libc::c_char,
        _: libc::c_ulong,
    ) -> libc::c_int;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
    fn access(__name: *const libc::c_char, __type: libc::c_int) -> libc::c_int;
    fn close(__fd: libc::c_int) -> libc::c_int;
    fn stat(__file: *const libc::c_char, __buf: *mut stat) -> libc::c_int;
    fn mkdir(__path: *const libc::c_char, __mode: __mode_t) -> libc::c_int;
    fn opendir(__name: *const libc::c_char) -> *mut DIR;
    fn closedir(__dirp: *mut DIR) -> libc::c_int;
    fn readdir(__dirp: *mut DIR) -> *mut dirent;
    fn glob(
        __pattern: *const libc::c_char,
        __flags: libc::c_int,
        __errfunc: Option::<
            unsafe extern "C" fn(*const libc::c_char, libc::c_int) -> libc::c_int,
        >,
        __pglob: *mut glob_t,
    ) -> libc::c_int;
    fn globfree(__pglob: *mut glob_t);
}
pub type size_t = libc::c_ulong;
pub type __uint16_t = libc::c_ushort;
pub type __uint32_t = libc::c_uint;
pub type __int64_t = libc::c_long;
pub type __uint64_t = libc::c_ulong;
pub type __dev_t = libc::c_ulong;
pub type __uid_t = libc::c_uint;
pub type __gid_t = libc::c_uint;
pub type __ino_t = libc::c_ulong;
pub type __mode_t = libc::c_uint;
pub type __nlink_t = libc::c_ulong;
pub type __off_t = libc::c_long;
pub type __off64_t = libc::c_long;
pub type __time_t = libc::c_long;
pub type __clockid_t = libc::c_int;
pub type __blksize_t = libc::c_long;
pub type __blkcnt_t = libc::c_long;
pub type __syscall_slong_t = libc::c_long;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _IO_FILE {
    pub _flags: libc::c_int,
    pub _IO_read_ptr: *mut libc::c_char,
    pub _IO_read_end: *mut libc::c_char,
    pub _IO_read_base: *mut libc::c_char,
    pub _IO_write_base: *mut libc::c_char,
    pub _IO_write_ptr: *mut libc::c_char,
    pub _IO_write_end: *mut libc::c_char,
    pub _IO_buf_base: *mut libc::c_char,
    pub _IO_buf_end: *mut libc::c_char,
    pub _IO_save_base: *mut libc::c_char,
    pub _IO_backup_base: *mut libc::c_char,
    pub _IO_save_end: *mut libc::c_char,
    pub _markers: *mut _IO_marker,
    pub _chain: *mut _IO_FILE,
    pub _fileno: libc::c_int,
    pub _flags2: libc::c_int,
    pub _old_offset: __off_t,
    pub _cur_column: libc::c_ushort,
    pub _vtable_offset: libc::c_schar,
    pub _shortbuf: [libc::c_char; 1],
    pub _lock: *mut libc::c_void,
    pub _offset: __off64_t,
    pub _codecvt: *mut _IO_codecvt,
    pub _wide_data: *mut _IO_wide_data,
    pub _freeres_list: *mut _IO_FILE,
    pub _freeres_buf: *mut libc::c_void,
    pub __pad5: size_t,
    pub _mode: libc::c_int,
    pub _unused2: [libc::c_char; 20],
}
pub type _IO_lock_t = ();
pub type FILE = _IO_FILE;
pub type clockid_t = __clockid_t;
pub type int64_t = __int64_t;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct timespec {
    pub tv_sec: __time_t,
    pub tv_nsec: __syscall_slong_t,
}
pub type C2RustUnnamed = libc::c_uint;
pub const _ISalnum: C2RustUnnamed = 8;
pub const _ISpunct: C2RustUnnamed = 4;
pub const _IScntrl: C2RustUnnamed = 2;
pub const _ISblank: C2RustUnnamed = 1;
pub const _ISgraph: C2RustUnnamed = 32768;
pub const _ISprint: C2RustUnnamed = 16384;
pub const _ISspace: C2RustUnnamed = 8192;
pub const _ISxdigit: C2RustUnnamed = 4096;
pub const _ISdigit: C2RustUnnamed = 2048;
pub const _ISalpha: C2RustUnnamed = 1024;
pub const _ISlower: C2RustUnnamed = 512;
pub const _ISupper: C2RustUnnamed = 256;
pub type uint16_t = __uint16_t;
pub type uint32_t = __uint32_t;
pub type uint64_t = __uint64_t;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct mt19937_state {
    pub seed_: libc::c_ulonglong,
    pub left_: libc::c_int,
    pub next_: libc::c_uint,
    pub state_: [libc::c_uint; 624],
    pub MATRIX_A: [libc::c_uint; 2],
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct stat {
    pub st_dev: __dev_t,
    pub st_ino: __ino_t,
    pub st_nlink: __nlink_t,
    pub st_mode: __mode_t,
    pub st_uid: __uid_t,
    pub st_gid: __gid_t,
    pub __pad0: libc::c_int,
    pub st_rdev: __dev_t,
    pub st_size: __off_t,
    pub st_blksize: __blksize_t,
    pub st_blocks: __blkcnt_t,
    pub st_atim: timespec,
    pub st_mtim: timespec,
    pub st_ctim: timespec,
    pub __glibc_reserved: [__syscall_slong_t; 3],
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct dirent {
    pub d_ino: __ino_t,
    pub d_off: __off_t,
    pub d_reclen: libc::c_ushort,
    pub d_type: libc::c_uchar,
    pub d_name: [libc::c_char; 256],
}
pub type DIR = __dirstream;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Tokenizer {
    pub vocab_size: uint32_t,
    pub token_table: *mut *mut libc::c_char,
    pub init_ok: libc::c_int,
    pub eot_token: libc::c_int,
}
pub type __size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct glob_t {
    pub gl_pathc: __size_t,
    pub gl_pathv: *mut *mut libc::c_char,
    pub gl_offs: __size_t,
    pub gl_flags: libc::c_int,
    pub gl_closedir: Option::<unsafe extern "C" fn(*mut libc::c_void) -> ()>,
    pub gl_readdir: Option::<
        unsafe extern "C" fn(*mut libc::c_void) -> *mut libc::c_void,
    >,
    pub gl_opendir: Option::<
        unsafe extern "C" fn(*const libc::c_char) -> *mut libc::c_void,
    >,
    pub gl_lstat: Option::<
        unsafe extern "C" fn(*const libc::c_char, *mut libc::c_void) -> libc::c_int,
    >,
    pub gl_stat: Option::<
        unsafe extern "C" fn(*const libc::c_char, *mut libc::c_void) -> libc::c_int,
    >,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct DataLoader {
    pub process_rank: libc::c_int,
    pub num_processes: libc::c_int,
    pub B: size_t,
    pub T: size_t,
    pub num_tokens: size_t,
    pub shard_num_samples: size_t,
    pub glob_result: glob_t,
    pub current_shard_idx: size_t,
    pub current_sample_idx: size_t,
    pub tokens_file: *mut FILE,
    pub buffer: *mut uint16_t,
    pub inputs: *mut libc::c_int,
    pub targets: *mut libc::c_int,
    pub shuffle_rng: mt19937_state,
    pub should_shuffle: libc::c_int,
    pub shard_indices: *mut libc::c_int,
    pub intra_shard_indices: *mut libc::c_int,
    pub total_batch_size_bytes: size_t,
    pub local_batch_offset_bytes: size_t,
    pub header_bytes: size_t,
    pub file_size_bytes: int64_t,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct EvalLoader {
    pub process_rank: libc::c_int,
    pub num_processes: libc::c_int,
    pub B: size_t,
    pub T: size_t,
    pub eval_file: *mut FILE,
    pub buffer: *mut uint16_t,
    pub num_examples: libc::c_int,
    pub num_batches: libc::c_int,
    pub start_example_index: libc::c_int,
    pub end_example_index: libc::c_int,
    pub current_example_index: libc::c_int,
    pub inputs: *mut libc::c_int,
    pub targets: *mut libc::c_int,
    pub mask: *mut libc::c_char,
    pub label: *mut libc::c_int,
    pub num_completions: libc::c_int,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GPT2Config {
    pub max_seq_len: libc::c_int,
    pub vocab_size: libc::c_int,
    pub padded_vocab_size: libc::c_int,
    pub num_layers: libc::c_int,
    pub num_heads: libc::c_int,
    pub channels: libc::c_int,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ParameterTensors {
    pub wte: *mut f32,
    pub wpe: *mut f32,
    pub ln1w: *mut f32,
    pub ln1b: *mut f32,
    pub qkvw: *mut f32,
    pub qkvb: *mut f32,
    pub attprojw: *mut f32,
    pub attprojb: *mut f32,
    pub ln2w: *mut f32,
    pub ln2b: *mut f32,
    pub fcw: *mut f32,
    pub fcb: *mut f32,
    pub fcprojw: *mut f32,
    pub fcprojb: *mut f32,
    pub lnfw: *mut f32,
    pub lnfb: *mut f32,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ActivationTensors {
    pub encoded: *mut f32,
    pub ln1: *mut f32,
    pub ln1_mean: *mut f32,
    pub ln1_rstd: *mut f32,
    pub qkv: *mut f32,
    pub atty: *mut f32,
    pub preatt: *mut f32,
    pub att: *mut f32,
    pub attproj: *mut f32,
    pub residual2: *mut f32,
    pub ln2: *mut f32,
    pub ln2_mean: *mut f32,
    pub ln2_rstd: *mut f32,
    pub fch: *mut f32,
    pub fch_gelu: *mut f32,
    pub fcproj: *mut f32,
    pub residual3: *mut f32,
    pub lnf: *mut f32,
    pub lnf_mean: *mut f32,
    pub lnf_rstd: *mut f32,
    pub logits: *mut f32,
    pub probs: *mut f32,
    pub losses: *mut f32,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GPT2 {
    pub params: ParameterTensors,
    pub param_sizes: [size_t; 16],
    pub params_memory: *mut f32,
    pub act_sizes: [size_t; 23],
    pub acts: ActivationTensors,
    pub acts_memory: *mut f32,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GPT2Const {
    pub config: GPT2Config,
    pub grads: ParameterTensors,
    pub grads_memory: *mut f32,
    pub m_memory: *mut f32,
    pub v_memory: *mut f32,
    pub num_activations: size_t,
    pub grads_acts: ActivationTensors,
    pub grads_acts_memory: *mut f32,
    pub batch_size: libc::c_int,
    pub seq_len: libc::c_int,
    pub inputs: *mut libc::c_int,
    pub targets: *mut libc::c_int,
    pub mean_loss: f32,
    pub num_parameters: size_t,
}
#[no_mangle]
pub unsafe extern "C" fn manual_seed(
    mut state: *mut mt19937_state,
    mut seed: libc::c_uint,
) {
    (*state).MATRIX_A[0 as libc::c_int as usize] = 0 as libc::c_uint;
    (*state).MATRIX_A[1 as libc::c_int as usize] = 0x9908b0df as libc::c_uint;
    (*state).state_[0 as libc::c_int as usize] = seed & 0xffffffff as libc::c_uint;
    let mut j: libc::c_uint = 1 as libc::c_int as libc::c_uint;
    while j < 624 as libc::c_uint {
        (*state)
            .state_[j
            as usize] = (1812433253 as libc::c_int as libc::c_uint)
            .wrapping_mul(
                (*state)
                    .state_[j.wrapping_sub(1 as libc::c_int as libc::c_uint) as usize]
                    ^ (*state)
                        .state_[j.wrapping_sub(1 as libc::c_int as libc::c_uint)
                        as usize] >> 30 as libc::c_int,
            )
            .wrapping_add(j);
        (*state).state_[j as usize] &= 0xffffffff as libc::c_uint;
        j = j.wrapping_add(1);
        j;
    }
    (*state).left_ = 1 as libc::c_int;
    (*state).next_ = 0 as libc::c_int as libc::c_uint;
}
#[no_mangle]
pub unsafe extern "C" fn next_state(mut state: *mut mt19937_state) {
    (*state).left_ = 624 as libc::c_uint as libc::c_int;
    (*state).next_ = 0 as libc::c_int as libc::c_uint;
    let mut y: libc::c_uint = 0;
    let mut j: libc::c_uint = 0;
    j = 0 as libc::c_int as libc::c_uint;
    while j < (624 as libc::c_uint).wrapping_sub(397 as libc::c_uint) {
        y = ((*state).state_[j as usize] as libc::c_ulong & 0x80000000 as libc::c_ulong
            | (*state).state_[j.wrapping_add(1 as libc::c_int as libc::c_uint) as usize]
                as libc::c_ulong & 0x7fffffff as libc::c_ulong) as libc::c_uint;
        (*state)
            .state_[j
            as usize] = (*state).state_[j.wrapping_add(397 as libc::c_uint) as usize]
            ^ y >> 1 as libc::c_int
            ^ (*state).MATRIX_A[(y & 0x1 as libc::c_int as libc::c_uint) as usize];
        j = j.wrapping_add(1);
        j;
    }
    while j < (624 as libc::c_uint).wrapping_sub(1 as libc::c_int as libc::c_uint) {
        y = ((*state).state_[j as usize] as libc::c_ulong & 0x80000000 as libc::c_ulong
            | (*state).state_[j.wrapping_add(1 as libc::c_int as libc::c_uint) as usize]
                as libc::c_ulong & 0x7fffffff as libc::c_ulong) as libc::c_uint;
        (*state)
            .state_[j
            as usize] = (*state)
            .state_[j
            .wrapping_add((397 as libc::c_uint).wrapping_sub(624 as libc::c_uint))
            as usize] ^ y >> 1 as libc::c_int
            ^ (*state).MATRIX_A[(y & 0x1 as libc::c_int as libc::c_uint) as usize];
        j = j.wrapping_add(1);
        j;
    }
    y = ((*state)
        .state_[(624 as libc::c_uint).wrapping_sub(1 as libc::c_int as libc::c_uint)
        as usize] as libc::c_ulong & 0x80000000 as libc::c_ulong
        | (*state).state_[0 as libc::c_int as usize] as libc::c_ulong
            & 0x7fffffff as libc::c_ulong) as libc::c_uint;
    (*state)
        .state_[(624 as libc::c_uint).wrapping_sub(1 as libc::c_int as libc::c_uint)
        as usize] = (*state)
        .state_[(397 as libc::c_uint).wrapping_sub(1 as libc::c_int as libc::c_uint)
        as usize] ^ y >> 1 as libc::c_int
        ^ (*state).MATRIX_A[(y & 0x1 as libc::c_int as libc::c_uint) as usize];
}
#[no_mangle]
pub unsafe extern "C" fn randint32(mut state: *mut mt19937_state) -> libc::c_uint {
    if state.is_null() {
        return 0 as libc::c_int as libc::c_uint;
    }
    if (*state).MATRIX_A[0 as libc::c_int as usize] != 0 as libc::c_int as libc::c_uint
        || (*state).MATRIX_A[1 as libc::c_int as usize] != 0x9908b0df as libc::c_uint
    {
        manual_seed(state, 5489 as libc::c_int as libc::c_uint);
    }
    (*state).left_ -= 1;
    if (*state).left_ <= 0 as libc::c_int {
        next_state(state);
    }
    let fresh0 = (*state).next_;
    (*state).next_ = ((*state).next_).wrapping_add(1);
    let mut y: libc::c_uint = (*state).state_[fresh0 as usize];
    y ^= y >> 11 as libc::c_int;
    y ^= y << 7 as libc::c_int & 0x9d2c5680 as libc::c_uint;
    y ^= y << 15 as libc::c_int & 0xefc60000 as libc::c_uint;
    y ^= y >> 18 as libc::c_int;
    return y;
}
#[no_mangle]
pub unsafe extern "C" fn randint64(mut state: *mut mt19937_state) -> libc::c_ulonglong {
    return (randint32(state) as libc::c_ulonglong) << 32 as libc::c_int
        | randint32(state) as libc::c_ulonglong;
}
#[no_mangle]
pub unsafe extern "C" fn randfloat32(mut state: *mut mt19937_state) -> f32 {
    return (randint32(state) as libc::c_ulonglong
        & ((1 as libc::c_ulonglong) << 24 as libc::c_int)
            .wrapping_sub(1 as libc::c_int as libc::c_ulonglong)) as f32
        * (1.0f32 / ((1 as libc::c_ulonglong) << 24 as libc::c_int) as f32);
}
#[no_mangle]
pub unsafe extern "C" fn randfloat64(mut state: *mut mt19937_state) -> libc::c_double {
    return (randint64(state)
        & ((1 as libc::c_ulonglong) << 53 as libc::c_int)
            .wrapping_sub(1 as libc::c_int as libc::c_ulonglong)) as libc::c_double
        * (1.0f64 / ((1 as libc::c_ulonglong) << 53 as libc::c_int) as libc::c_double);
}
#[no_mangle]
pub unsafe extern "C" fn uniform_(
    mut data: *mut f32,
    mut numel: libc::c_uint,
    mut from: f32,
    mut to: f32,
    mut state: *mut mt19937_state,
) {
    let mut t: libc::c_uint = 0 as libc::c_int as libc::c_uint;
    while t < numel {
        *data.offset(t as isize) = randfloat32(state) * (to - from) + from;
        t = t.wrapping_add(1);
        t;
    }
}
#[no_mangle]
pub unsafe extern "C" fn normal_fill_16(
    mut data: *mut f32,
    mut mean: f32,
    mut std: f32,
) {
    let mut t: libc::c_uint = 0 as libc::c_int as libc::c_uint;
    while t < 8 as libc::c_int as libc::c_uint {
        let mut u1: f32 = 1 as libc::c_int as f32
            - *data.offset(t as isize);
        let mut u2: f32 = *data
            .offset(t.wrapping_add(8 as libc::c_int as libc::c_uint) as isize);
        let mut radius: f32 = sqrtf(
            -(2 as libc::c_int) as f32 * logf(u1 + 1e-12f32),
        );
        let mut theta: f32 = (2.0f64 * 3.14159265358979323846f64
            * u2 as libc::c_double) as f32;
        *data.offset(t as isize) = radius * cosf(theta) * std + mean;
        *data
            .offset(
                t.wrapping_add(8 as libc::c_int as libc::c_uint) as isize,
            ) = radius * sinf(theta) * std + mean;
        t = t.wrapping_add(1);
        t;
    }
}
#[no_mangle]
pub unsafe extern "C" fn normal_fill(
    mut data: *mut f32,
    mut numel: libc::c_uint,
    mut mean: f32,
    mut std: f32,
    mut state: *mut mt19937_state,
) {
    let mut t: libc::c_uint = 0 as libc::c_int as libc::c_uint;
    while t < numel {
        *data.offset(t as isize) = randfloat32(state);
        t = t.wrapping_add(1);
        t;
    }
    let mut i: libc::c_uint = 0 as libc::c_int as libc::c_uint;
    while i < numel.wrapping_sub(15 as libc::c_int as libc::c_uint) {
        normal_fill_16(data.offset(i as isize), mean, std);
        i = i.wrapping_add(16 as libc::c_int as libc::c_uint);
    }
    if numel.wrapping_rem(16 as libc::c_int as libc::c_uint)
        != 0 as libc::c_int as libc::c_uint
    {
        data = data.offset(numel as isize).offset(-(16 as libc::c_int as isize));
        let mut i_0: libc::c_uint = 0 as libc::c_int as libc::c_uint;
        while i_0 < 16 as libc::c_int as libc::c_uint {
            *data.offset(i_0 as isize) = randfloat32(state);
            i_0 = i_0.wrapping_add(1);
            i_0;
        }
        normal_fill_16(data, mean, std);
    }
}
#[no_mangle]
pub unsafe extern "C" fn normal_(
    mut data: *mut f32,
    mut numel: libc::c_uint,
    mut mean: f32,
    mut std: f32,
    mut state: *mut mt19937_state,
) {
    if numel >= 16 as libc::c_int as libc::c_uint {
        normal_fill(data, numel, mean, std, state);
    } else {
        let mut next_double_normal_sample: libc::c_double = 0.0f64;
        let mut has_next_double_normal_sample: libc::c_int = 0 as libc::c_int;
        let mut t: libc::c_uint = 0 as libc::c_int as libc::c_uint;
        while t < numel {
            if has_next_double_normal_sample != 0 {
                *data
                    .offset(
                        t as isize,
                    ) = (next_double_normal_sample * std as libc::c_double
                    + mean as libc::c_double) as f32;
                has_next_double_normal_sample = 0 as libc::c_int;
            } else {
                let mut u1: f32 = randfloat64(state) as f32;
                let mut u2: f32 = randfloat64(state) as f32;
                let mut radius: f32 = sqrtf(
                    -(2 as libc::c_int) as f32
                        * logf(1 as libc::c_int as f32 - u2 + 1e-12f32),
                );
                let mut theta: f32 = (2.0f64 * 3.14159265358979323846f64
                    * u1 as libc::c_double) as f32;
                next_double_normal_sample = (radius * sinf(theta)) as libc::c_double;
                has_next_double_normal_sample = 1 as libc::c_int;
                *data.offset(t as isize) = radius * cosf(theta) * std + mean;
            }
            t = t.wrapping_add(1);
            t;
        }
    };
}
#[no_mangle]
pub unsafe extern "C" fn init_identity_permutation(
    mut data: *mut libc::c_int,
    mut numel: libc::c_int,
) {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < numel {
        *data.offset(i as isize) = i;
        i += 1;
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn random_permutation(
    mut data: *mut libc::c_int,
    mut numel: libc::c_int,
    mut state: *mut mt19937_state,
) {
    let mut i: libc::c_int = numel - 1 as libc::c_int;
    while i > 0 as libc::c_int {
        let mut j: libc::c_int = (randint32(state))
            .wrapping_rem((i + 1 as libc::c_int) as libc::c_uint) as libc::c_int;
        let mut tmp: libc::c_int = *data.offset(i as isize);
        *data.offset(i as isize) = *data.offset(j as isize);
        *data.offset(j as isize) = tmp;
        i -= 1;
        i;
    }
}
#[no_mangle]
#[inline]
#[linkage = "external"]
pub unsafe extern "C" fn fopen_check(
    mut path: *const libc::c_char,
    mut mode: *const libc::c_char,
    mut file: *const libc::c_char,
    mut line: libc::c_int,
) -> *mut FILE {
    let mut fp: *mut FILE = fopen(path, mode);
    if fp.is_null() {
        fprintf(
            stderr,
            b"Error: Failed to open file '%s' at %s:%d\n\0" as *const u8
                as *const libc::c_char,
            path,
            file,
            line,
        );
        fprintf(stderr, b"Error details:\n\0" as *const u8 as *const libc::c_char);
        fprintf(stderr, b"  File: %s\n\0" as *const u8 as *const libc::c_char, file);
        fprintf(stderr, b"  Line: %d\n\0" as *const u8 as *const libc::c_char, line);
        fprintf(stderr, b"  Path: %s\n\0" as *const u8 as *const libc::c_char, path);
        fprintf(stderr, b"  Mode: %s\n\0" as *const u8 as *const libc::c_char, mode);
        fprintf(
            stderr,
            b"---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n\0"
                as *const u8 as *const libc::c_char,
        );
        fprintf(
            stderr,
            b"---> HINT 2: possibly try to re-run `python train_gpt2.py`\n\0"
                as *const u8 as *const libc::c_char,
        );
        exit(1 as libc::c_int);
    }
    return fp;
}
#[no_mangle]
#[inline]
#[linkage = "external"]
pub unsafe extern "C" fn fread_check(
    mut ptr: *mut libc::c_void,
    mut size: size_t,
    mut nmemb: size_t,
    mut stream: *mut FILE,
    mut file: *const libc::c_char,
    mut line: libc::c_int,
) {
    let mut result: size_t = fread(ptr, size, nmemb, stream);
    if result != nmemb {
        if feof(stream) != 0 {
            fprintf(
                stderr,
                b"Error: Unexpected end of file at %s:%d\n\0" as *const u8
                    as *const libc::c_char,
                file,
                line,
            );
        } else if ferror(stream) != 0 {
            fprintf(
                stderr,
                b"Error: File read error at %s:%d\n\0" as *const u8
                    as *const libc::c_char,
                file,
                line,
            );
        } else {
            fprintf(
                stderr,
                b"Error: Partial read at %s:%d. Expected %zu elements, read %zu\n\0"
                    as *const u8 as *const libc::c_char,
                file,
                line,
                nmemb,
                result,
            );
        }
        fprintf(stderr, b"Error details:\n\0" as *const u8 as *const libc::c_char);
        fprintf(stderr, b"  File: %s\n\0" as *const u8 as *const libc::c_char, file);
        fprintf(stderr, b"  Line: %d\n\0" as *const u8 as *const libc::c_char, line);
        fprintf(
            stderr,
            b"  Expected elements: %zu\n\0" as *const u8 as *const libc::c_char,
            nmemb,
        );
        fprintf(
            stderr,
            b"  Read elements: %zu\n\0" as *const u8 as *const libc::c_char,
            result,
        );
        exit(1 as libc::c_int);
    }
}
#[no_mangle]
#[inline]
#[linkage = "external"]
pub unsafe extern "C" fn fclose_check(
    mut fp: *mut FILE,
    mut file: *const libc::c_char,
    mut line: libc::c_int,
) {
    if fclose(fp) != 0 as libc::c_int {
        fprintf(
            stderr,
            b"Error: Failed to close file at %s:%d\n\0" as *const u8
                as *const libc::c_char,
            file,
            line,
        );
        fprintf(stderr, b"Error details:\n\0" as *const u8 as *const libc::c_char);
        fprintf(stderr, b"  File: %s\n\0" as *const u8 as *const libc::c_char, file);
        fprintf(stderr, b"  Line: %d\n\0" as *const u8 as *const libc::c_char, line);
        exit(1 as libc::c_int);
    }
}
#[no_mangle]
#[inline]
#[linkage = "external"]
pub unsafe extern "C" fn sclose_check(
    mut sockfd: libc::c_int,
    mut file: *const libc::c_char,
    mut line: libc::c_int,
) {
    if close(sockfd) != 0 as libc::c_int {
        fprintf(
            stderr,
            b"Error: Failed to close socket at %s:%d\n\0" as *const u8
                as *const libc::c_char,
            file,
            line,
        );
        fprintf(stderr, b"Error details:\n\0" as *const u8 as *const libc::c_char);
        fprintf(stderr, b"  File: %s\n\0" as *const u8 as *const libc::c_char, file);
        fprintf(stderr, b"  Line: %d\n\0" as *const u8 as *const libc::c_char, line);
        exit(1 as libc::c_int);
    }
}
#[no_mangle]
#[inline]
#[linkage = "external"]
pub unsafe extern "C" fn fseek_check(
    mut fp: *mut FILE,
    mut off: libc::c_long,
    mut whence: libc::c_int,
    mut file: *const libc::c_char,
    mut line: libc::c_int,
) {
    if fseek(fp, off, whence) != 0 as libc::c_int {
        fprintf(
            stderr,
            b"Error: Failed to seek in file at %s:%d\n\0" as *const u8
                as *const libc::c_char,
            file,
            line,
        );
        fprintf(stderr, b"Error details:\n\0" as *const u8 as *const libc::c_char);
        fprintf(stderr, b"  Offset: %ld\n\0" as *const u8 as *const libc::c_char, off);
        fprintf(stderr, b"  Whence: %d\n\0" as *const u8 as *const libc::c_char, whence);
        fprintf(stderr, b"  File:   %s\n\0" as *const u8 as *const libc::c_char, file);
        fprintf(stderr, b"  Line:   %d\n\0" as *const u8 as *const libc::c_char, line);
        exit(1 as libc::c_int);
    }
}
#[no_mangle]
#[inline]
#[linkage = "external"]
pub unsafe extern "C" fn fwrite_check(
    mut ptr: *mut libc::c_void,
    mut size: size_t,
    mut nmemb: size_t,
    mut stream: *mut FILE,
    mut file: *const libc::c_char,
    mut line: libc::c_int,
) {
    let mut result: size_t = fwrite(ptr, size, nmemb, stream);
    if result != nmemb {
        if feof(stream) != 0 {
            fprintf(
                stderr,
                b"Error: Unexpected end of file at %s:%d\n\0" as *const u8
                    as *const libc::c_char,
                file,
                line,
            );
        } else if ferror(stream) != 0 {
            fprintf(
                stderr,
                b"Error: File write error at %s:%d\n\0" as *const u8
                    as *const libc::c_char,
                file,
                line,
            );
        } else {
            fprintf(
                stderr,
                b"Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n\0"
                    as *const u8 as *const libc::c_char,
                file,
                line,
                nmemb,
                result,
            );
        }
        fprintf(stderr, b"Error details:\n\0" as *const u8 as *const libc::c_char);
        fprintf(stderr, b"  File: %s\n\0" as *const u8 as *const libc::c_char, file);
        fprintf(stderr, b"  Line: %d\n\0" as *const u8 as *const libc::c_char, line);
        fprintf(
            stderr,
            b"  Expected elements: %zu\n\0" as *const u8 as *const libc::c_char,
            nmemb,
        );
        fprintf(
            stderr,
            b"  Written elements: %zu\n\0" as *const u8 as *const libc::c_char,
            result,
        );
        exit(1 as libc::c_int);
    }
}
#[no_mangle]
#[inline]
#[linkage = "external"]
pub unsafe extern "C" fn malloc_check(
    mut size: size_t,
    mut file: *const libc::c_char,
    mut line: libc::c_int,
) -> *mut libc::c_void {
    let mut ptr: *mut libc::c_void = malloc(size);
    if ptr.is_null() {
        fprintf(
            stderr,
            b"Error: Memory allocation failed at %s:%d\n\0" as *const u8
                as *const libc::c_char,
            file,
            line,
        );
        fprintf(stderr, b"Error details:\n\0" as *const u8 as *const libc::c_char);
        fprintf(stderr, b"  File: %s\n\0" as *const u8 as *const libc::c_char, file);
        fprintf(stderr, b"  Line: %d\n\0" as *const u8 as *const libc::c_char, line);
        fprintf(
            stderr,
            b"  Size: %zu bytes\n\0" as *const u8 as *const libc::c_char,
            size,
        );
        exit(1 as libc::c_int);
    }
    return ptr;
}
#[no_mangle]
#[inline]
#[linkage = "external"]
pub unsafe extern "C" fn token_check(
    mut tokens: *const libc::c_int,
    mut token_count: libc::c_int,
    mut vocab_size: libc::c_int,
    mut file: *const libc::c_char,
    mut line: libc::c_int,
) {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < token_count {
        if !(0 as libc::c_int <= *tokens.offset(i as isize)
            && *tokens.offset(i as isize) < vocab_size)
        {
            fprintf(
                stderr,
                b"Error: Token out of vocabulary at %s:%d\n\0" as *const u8
                    as *const libc::c_char,
                file,
                line,
            );
            fprintf(stderr, b"Error details:\n\0" as *const u8 as *const libc::c_char);
            fprintf(stderr, b"  File: %s\n\0" as *const u8 as *const libc::c_char, file);
            fprintf(stderr, b"  Line: %d\n\0" as *const u8 as *const libc::c_char, line);
            fprintf(
                stderr,
                b"  Token: %d\n\0" as *const u8 as *const libc::c_char,
                *tokens.offset(i as isize),
            );
            fprintf(
                stderr,
                b"  Position: %d\n\0" as *const u8 as *const libc::c_char,
                i,
            );
            fprintf(
                stderr,
                b"  Vocab: %d\n\0" as *const u8 as *const libc::c_char,
                vocab_size,
            );
            exit(1 as libc::c_int);
        }
        i += 1;
        i;
    }
}
#[no_mangle]
#[inline]
#[linkage = "external"]
pub unsafe extern "C" fn create_dir_if_not_exists(mut dir: *const libc::c_char) {
    if dir.is_null() {
        return;
    }
    let mut st: stat = {
        let mut init = stat {
            st_dev: 0 as libc::c_int as __dev_t,
            st_ino: 0,
            st_nlink: 0,
            st_mode: 0,
            st_uid: 0,
            st_gid: 0,
            __pad0: 0,
            st_rdev: 0,
            st_size: 0,
            st_blksize: 0,
            st_blocks: 0,
            st_atim: timespec { tv_sec: 0, tv_nsec: 0 },
            st_mtim: timespec { tv_sec: 0, tv_nsec: 0 },
            st_ctim: timespec { tv_sec: 0, tv_nsec: 0 },
            __glibc_reserved: [0; 3],
        };
        init
    };
    if stat(dir, &mut st) == -(1 as libc::c_int) {
        if mkdir(dir, 0o700 as libc::c_int as __mode_t) == -(1 as libc::c_int) {
            printf(
                b"ERROR: could not create directory: %s\n\0" as *const u8
                    as *const libc::c_char,
                dir,
            );
            exit(1 as libc::c_int);
        }
        printf(b"created directory: %s\n\0" as *const u8 as *const libc::c_char, dir);
    }
}
#[no_mangle]
#[inline]
#[linkage = "external"]
pub unsafe extern "C" fn find_max_step(
    mut output_log_dir: *const libc::c_char,
) -> libc::c_int {
    if output_log_dir.is_null() {
        return -(1 as libc::c_int);
    }
    let mut dir: *mut DIR = 0 as *mut DIR;
    let mut entry: *mut dirent = 0 as *mut dirent;
    let mut max_step: libc::c_int = -(1 as libc::c_int);
    dir = opendir(output_log_dir);
    if dir.is_null() {
        return -(1 as libc::c_int);
    }
    loop {
        entry = readdir(dir);
        if entry.is_null() {
            break;
        }
        if strncmp(
            ((*entry).d_name).as_mut_ptr(),
            b"DONE_\0" as *const u8 as *const libc::c_char,
            5 as libc::c_int as libc::c_ulong,
        ) == 0 as libc::c_int
        {
            let mut step: libc::c_int = atoi(
                ((*entry).d_name).as_mut_ptr().offset(5 as libc::c_int as isize),
            );
            if step > max_step {
                max_step = step;
            }
        }
    }
    closedir(dir);
    return max_step;
}
#[no_mangle]
#[inline]
#[linkage = "external"]
pub unsafe extern "C" fn ends_with_bin(mut str: *const libc::c_char) -> libc::c_int {
    if str.is_null() {
        return 0 as libc::c_int;
    }
    let mut len: size_t = strlen(str);
    let mut suffix: *const libc::c_char = b".bin\0" as *const u8 as *const libc::c_char;
    let mut suffix_len: size_t = strlen(suffix);
    if len < suffix_len {
        return 0 as libc::c_int;
    }
    let mut suffix_matches: libc::c_int = (strncmp(
        str.offset(len as isize).offset(-(suffix_len as isize)),
        suffix,
        suffix_len,
    ) == 0 as libc::c_int) as libc::c_int;
    return suffix_matches;
}
#[no_mangle]
pub unsafe extern "C" fn safe_printf(mut piece: *const libc::c_char) {
    if piece.is_null() {
        return;
    }
    if *piece.offset(0 as libc::c_int as isize) as libc::c_int == '\0' as i32 {
        return;
    }
    if *piece.offset(1 as libc::c_int as isize) as libc::c_int == '\0' as i32 {
        let mut byte_val: libc::c_uchar = *piece.offset(0 as libc::c_int as isize)
            as libc::c_uchar;
        if !(*(*__ctype_b_loc()).offset(byte_val as libc::c_int as isize) as libc::c_int
            & _ISprint as libc::c_int as libc::c_ushort as libc::c_int != 0
            || *(*__ctype_b_loc()).offset(byte_val as libc::c_int as isize)
                as libc::c_int & _ISspace as libc::c_int as libc::c_ushort as libc::c_int
                != 0)
        {
            return;
        }
    }
    printf(b"%s\0" as *const u8 as *const libc::c_char, piece);
}
#[no_mangle]
pub unsafe extern "C" fn tokenizer_init(
    mut tokenizer: *mut Tokenizer,
    mut filename: *const libc::c_char,
) {
    let mut file: *mut FILE = fopen(
        filename,
        b"rb\0" as *const u8 as *const libc::c_char,
    );
    if file.is_null() {
        printf(b"---\n\0" as *const u8 as *const libc::c_char);
        printf(
            b"WARNING: Failed to open the tokenizer file %s\n\0" as *const u8
                as *const libc::c_char,
            filename,
        );
        printf(
            b"The Tokenizer is a new feature added April 14 2024.\n\0" as *const u8
                as *const libc::c_char,
        );
        printf(
            b"Re-run `python train_gpt2.py` to write it\n\0" as *const u8
                as *const libc::c_char,
        );
        printf(b"---\n\0" as *const u8 as *const libc::c_char);
        (*tokenizer).init_ok = 0 as libc::c_int;
        return;
    }
    let mut header: [uint32_t; 256] = [0; 256];
    fread_check(
        header.as_mut_ptr() as *mut libc::c_void,
        ::core::mem::size_of::<uint32_t>() as libc::c_ulong,
        256 as libc::c_int as size_t,
        file,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        542 as libc::c_int,
    );
    let mut version: libc::c_int = header[1 as libc::c_int as usize] as libc::c_int;
    (*tokenizer).vocab_size = header[2 as libc::c_int as usize];
    if version == 1 as libc::c_int {
        (*tokenizer).eot_token = 50256 as libc::c_int;
    } else if version == 2 as libc::c_int {
        (*tokenizer).eot_token = header[3 as libc::c_int as usize] as libc::c_int;
    } else {
        fprintf(
            stderr,
            b"Tokenizer model file %s has bad version: %d\n\0" as *const u8
                as *const libc::c_char,
            filename,
            version,
        );
        exit(1 as libc::c_int);
    }
    let mut length: libc::c_uchar = 0;
    (*tokenizer)
        .token_table = malloc_check(
        ((*tokenizer).vocab_size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong),
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        559 as libc::c_int,
    ) as *mut *mut libc::c_char;
    let mut i: uint32_t = 0 as libc::c_int as uint32_t;
    while i < (*tokenizer).vocab_size {
        fread_check(
            &mut length as *mut libc::c_uchar as *mut libc::c_void,
            ::core::mem::size_of::<libc::c_uchar>() as libc::c_ulong,
            1 as libc::c_int as size_t,
            file,
            b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
            561 as libc::c_int,
        );
        let mut token_bytes: *mut libc::c_char = malloc_check(
            (length as libc::c_int + 1 as libc::c_int) as size_t,
            b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
            563 as libc::c_int,
        ) as *mut libc::c_char;
        fread_check(
            token_bytes as *mut libc::c_void,
            ::core::mem::size_of::<libc::c_char>() as libc::c_ulong,
            length as size_t,
            file,
            b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
            564 as libc::c_int,
        );
        *token_bytes.offset(length as isize) = '\0' as i32 as libc::c_char;
        let ref mut fresh1 = *((*tokenizer).token_table).offset(i as isize);
        *fresh1 = token_bytes;
        i = i.wrapping_add(1);
        i;
    }
    fclose_check(
        file,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        569 as libc::c_int,
    );
    (*tokenizer).init_ok = 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn tokenizer_decode(
    mut tokenizer: *mut Tokenizer,
    mut token_id: uint32_t,
) -> *const libc::c_char {
    if (*tokenizer).init_ok == 0 as libc::c_int {
        return 0 as *const libc::c_char;
    }
    if token_id < (*tokenizer).vocab_size {
        return *((*tokenizer).token_table).offset(token_id as isize)
    } else {
        printf(
            b"invalid token id %u!\n\0" as *const u8 as *const libc::c_char,
            token_id,
        );
        return 0 as *const libc::c_char;
    };
}
#[no_mangle]
pub unsafe extern "C" fn tokenizer_free(mut tokenizer: *mut Tokenizer) {
    if (*tokenizer).init_ok != 0 {
        let mut i: uint32_t = 0 as libc::c_int as uint32_t;
        while i < (*tokenizer).vocab_size {
            free(*((*tokenizer).token_table).offset(i as isize) as *mut libc::c_void);
            i = i.wrapping_add(1);
            i;
        }
        free((*tokenizer).token_table as *mut libc::c_void);
    }
}
#[no_mangle]
pub unsafe extern "C" fn dataloader_load_shard_(
    mut loader: *mut DataLoader,
    mut shard_index: libc::c_int,
) -> int64_t {
    if (*loader).should_shuffle != 0 {
        shard_index = *((*loader).shard_indices).offset(shard_index as isize);
    }
    let mut filename: *const libc::c_char = *((*loader).glob_result.gl_pathv)
        .offset(shard_index as isize);
    if !((*loader).tokens_file).is_null() {
        fclose_check(
            (*loader).tokens_file,
            b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
            660 as libc::c_int,
        );
    }
    (*loader)
        .tokens_file = fopen_check(
        filename,
        b"rb\0" as *const u8 as *const libc::c_char,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        662 as libc::c_int,
    );
    let mut header: [libc::c_int; 256] = [0; 256];
    fread_check(
        header.as_mut_ptr() as *mut libc::c_void,
        ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
        256 as libc::c_int as size_t,
        (*loader).tokens_file,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        665 as libc::c_int,
    );
    if header[0 as libc::c_int as usize] != 20240520 as libc::c_int {
        printf(b"Bad magic in the data file\n\0" as *const u8 as *const libc::c_char);
        printf(
            b"---> HINT: Are you passing in a correct file?\n\0" as *const u8
                as *const libc::c_char,
        );
        printf(
            b"---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n\0"
                as *const u8 as *const libc::c_char,
        );
        exit(1 as libc::c_int);
    }
    if header[1 as libc::c_int as usize] != 1 as libc::c_int {
        printf(b"Bad version in data file\n\0" as *const u8 as *const libc::c_char);
        exit(1 as libc::c_int);
    }
    let mut ntok: int64_t = header[2 as libc::c_int as usize] as int64_t;
    fseek_check(
        (*loader).tokens_file,
        0 as libc::c_int as libc::c_long,
        2 as libc::c_int,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        676 as libc::c_int,
    );
    (*loader).file_size_bytes = ftell((*loader).tokens_file);
    fseek_check(
        (*loader).tokens_file,
        0 as libc::c_int as libc::c_long,
        0 as libc::c_int,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        678 as libc::c_int,
    );
    let mut expected_file_size: int64_t = (256 as libc::c_int as libc::c_ulong)
        .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong)
        .wrapping_add(
            (ntok as libc::c_ulong)
                .wrapping_mul(::core::mem::size_of::<uint16_t>() as libc::c_ulong),
        ) as int64_t;
    if (*loader).file_size_bytes != expected_file_size {
        printf(
            b"Error: file size is not as expected\n\0" as *const u8
                as *const libc::c_char,
        );
        exit(1 as libc::c_int);
    }
    (*loader)
        .shard_num_samples = (ntok as libc::c_ulong)
        .wrapping_mul(::core::mem::size_of::<uint16_t>() as libc::c_ulong)
        .wrapping_sub(::core::mem::size_of::<uint16_t>() as libc::c_ulong)
        .wrapping_div((*loader).total_batch_size_bytes);
    return ntok;
}
#[no_mangle]
pub unsafe extern "C" fn prepare_intra_shard_indices_(mut loader: *mut DataLoader) {
    if !((*loader).intra_shard_indices).is_null() {
        free((*loader).intra_shard_indices as *mut libc::c_void);
    }
    (*loader)
        .intra_shard_indices = malloc_check(
        ((*loader).shard_num_samples)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        696 as libc::c_int,
    ) as *mut libc::c_int;
    init_identity_permutation(
        (*loader).intra_shard_indices,
        (*loader).shard_num_samples as libc::c_int,
    );
    random_permutation(
        (*loader).intra_shard_indices,
        (*loader).shard_num_samples as libc::c_int,
        &mut (*loader).shuffle_rng,
    );
}
#[no_mangle]
pub unsafe extern "C" fn dataloader_reset(mut loader: *mut DataLoader) {
    (*loader).current_shard_idx = 0 as libc::c_int as size_t;
    (*loader).current_sample_idx = 0 as libc::c_int as size_t;
    if (*loader).should_shuffle != 0 {
        random_permutation(
            (*loader).shard_indices,
            (*loader).glob_result.gl_pathc as libc::c_int,
            &mut (*loader).shuffle_rng,
        );
    }
    dataloader_load_shard_(loader, (*loader).current_shard_idx as libc::c_int);
    if (*loader).should_shuffle != 0 {
        prepare_intra_shard_indices_(loader);
    }
}
#[no_mangle]
pub unsafe extern "C" fn dataloader_advance_(mut loader: *mut DataLoader) {
    if (*loader).current_shard_idx
        == ((*loader).glob_result.gl_pathc)
            .wrapping_sub(1 as libc::c_int as libc::c_ulong)
    {
        dataloader_reset(loader);
        return;
    }
    (*loader)
        .current_shard_idx = ((*loader).current_shard_idx)
        .wrapping_add(1 as libc::c_int as libc::c_ulong)
        .wrapping_rem((*loader).glob_result.gl_pathc);
    (*loader).current_sample_idx = 0 as libc::c_int as size_t;
    dataloader_load_shard_(loader, (*loader).current_shard_idx as libc::c_int);
    if (*loader).should_shuffle != 0 {
        prepare_intra_shard_indices_(loader);
    }
}
#[no_mangle]
pub unsafe extern "C" fn dataloader_init(
    mut loader: *mut DataLoader,
    mut filename_pattern: *const libc::c_char,
    mut B: size_t,
    mut T: size_t,
    mut process_rank: libc::c_int,
    mut num_processes: libc::c_int,
    mut should_shuffle: libc::c_int,
) {
    (*loader).process_rank = process_rank;
    (*loader).num_processes = num_processes;
    (*loader).B = B;
    (*loader).T = T;
    (*loader).tokens_file = 0 as *mut FILE;
    (*loader).should_shuffle = should_shuffle;
    (*loader)
        .header_bytes = (256 as libc::c_int as libc::c_ulong)
        .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong);
    (*loader)
        .total_batch_size_bytes = ((*loader).num_processes as libc::c_ulong)
        .wrapping_mul(((*loader).B).wrapping_mul((*loader).T))
        .wrapping_mul(::core::mem::size_of::<uint16_t>() as libc::c_ulong);
    (*loader)
        .local_batch_offset_bytes = ((*loader).process_rank as libc::c_ulong)
        .wrapping_mul((*loader).B)
        .wrapping_mul((*loader).T)
        .wrapping_mul(::core::mem::size_of::<uint16_t>() as libc::c_ulong);
    let mut glob_status: libc::c_int = glob(
        filename_pattern,
        0 as libc::c_int,
        None,
        &mut (*loader).glob_result,
    );
    if glob_status != 0 as libc::c_int {
        printf(
            b"Error: failed to glob pattern: %s\n\0" as *const u8 as *const libc::c_char,
            filename_pattern,
        );
        exit(1 as libc::c_int);
    }
    if (*loader).glob_result.gl_pathc == 0 as libc::c_int as libc::c_ulong {
        printf(
            b"Error: no files found matching the pattern: %s\n\0" as *const u8
                as *const libc::c_char,
            filename_pattern,
        );
        exit(1 as libc::c_int);
    }
    if should_shuffle != 0 {
        let mut shuffle_rng: mt19937_state = mt19937_state {
            seed_: 0,
            left_: 0,
            next_: 0,
            state_: [0; 624],
            MATRIX_A: [0; 2],
        };
        manual_seed(
            &mut shuffle_rng,
            (42 as libc::c_int + process_rank) as libc::c_uint,
        );
        (*loader).shuffle_rng = shuffle_rng;
        (*loader)
            .shard_indices = malloc_check(
            ((*loader).glob_result.gl_pathc)
                .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
            b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
            765 as libc::c_int,
        ) as *mut libc::c_int;
        init_identity_permutation(
            (*loader).shard_indices,
            (*loader).glob_result.gl_pathc as libc::c_int,
        );
        (*loader).intra_shard_indices = 0 as *mut libc::c_int;
    }
    let mut ntok_total: int64_t = 0 as libc::c_int as int64_t;
    let mut shard_index: libc::c_int = 0 as libc::c_int;
    while (shard_index as libc::c_ulong) < (*loader).glob_result.gl_pathc {
        let mut shard_ntok: int64_t = dataloader_load_shard_(loader, shard_index);
        ntok_total += shard_ntok;
        shard_index += 1;
        shard_index;
    }
    (*loader)
        .buffer = malloc_check(
        B
            .wrapping_mul(T)
            .wrapping_add(1 as libc::c_int as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<uint16_t>() as libc::c_ulong),
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        785 as libc::c_int,
    ) as *mut uint16_t;
    (*loader)
        .inputs = malloc_check(
        B
            .wrapping_mul(T)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        786 as libc::c_int,
    ) as *mut libc::c_int;
    (*loader)
        .targets = malloc_check(
        B
            .wrapping_mul(T)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        787 as libc::c_int,
    ) as *mut libc::c_int;
    (*loader).num_tokens = ntok_total as size_t;
    dataloader_reset(loader);
}
#[no_mangle]
pub unsafe extern "C" fn dataloader_load_batch(mut loader: *mut DataLoader) {
    let mut idx: size_t = if (*loader).should_shuffle != 0 {
        *((*loader).intra_shard_indices).offset((*loader).current_sample_idx as isize)
            as libc::c_ulong
    } else {
        (*loader).current_sample_idx
    };
    let mut global_batch_offset_bytes: size_t = idx
        .wrapping_mul((*loader).total_batch_size_bytes);
    let mut current_offset: int64_t = ((*loader).header_bytes)
        .wrapping_add(global_batch_offset_bytes)
        .wrapping_add((*loader).local_batch_offset_bytes) as int64_t;
    let mut B: size_t = (*loader).B;
    let mut T: size_t = (*loader).T;
    fseek_check(
        (*loader).tokens_file,
        current_offset as libc::c_int as libc::c_long,
        0 as libc::c_int,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        804 as libc::c_int,
    );
    fread_check(
        (*loader).buffer as *mut libc::c_void,
        ::core::mem::size_of::<uint16_t>() as libc::c_ulong,
        B.wrapping_mul(T).wrapping_add(1 as libc::c_int as libc::c_ulong),
        (*loader).tokens_file,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        805 as libc::c_int,
    );
    let mut i: libc::c_int = 0 as libc::c_int;
    while (i as libc::c_ulong) < B.wrapping_mul(T) {
        *((*loader).inputs)
            .offset(i as isize) = *((*loader).buffer).offset(i as isize) as libc::c_int;
        *((*loader).targets)
            .offset(
                i as isize,
            ) = *((*loader).buffer).offset((i + 1 as libc::c_int) as isize)
            as libc::c_int;
        i += 1;
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn dataloader_next_batch(mut loader: *mut DataLoader) {
    if (*loader).current_sample_idx >= (*loader).shard_num_samples {
        dataloader_advance_(loader);
    }
    dataloader_load_batch(loader);
    (*loader)
        .current_sample_idx = ((*loader).current_sample_idx as libc::c_ulong)
        .wrapping_add(1 as libc::c_int as libc::c_ulong) as size_t as size_t;
}
#[no_mangle]
pub unsafe extern "C" fn dataloader_resume(
    mut loader: *mut DataLoader,
    mut current_shard_idx: size_t,
    mut current_sample_idx: size_t,
) {
    (*loader).current_shard_idx = current_shard_idx;
    (*loader).current_sample_idx = current_sample_idx;
    dataloader_load_shard_(loader, (*loader).current_shard_idx as libc::c_int);
}
#[no_mangle]
pub unsafe extern "C" fn dataloader_free(mut loader: *mut DataLoader) {
    free((*loader).buffer as *mut libc::c_void);
    free((*loader).inputs as *mut libc::c_void);
    free((*loader).targets as *mut libc::c_void);
    if (*loader).should_shuffle != 0 {
        free((*loader).shard_indices as *mut libc::c_void);
        free((*loader).intra_shard_indices as *mut libc::c_void);
    }
    fclose_check(
        (*loader).tokens_file,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        838 as libc::c_int,
    );
    globfree(&mut (*loader).glob_result);
}
#[no_mangle]
pub unsafe extern "C" fn evalloader_reset(mut loader: *mut EvalLoader) {
    let mut examples_per_process: libc::c_int = ((*loader).num_examples
        + (*loader).num_processes - 1 as libc::c_int) / (*loader).num_processes;
    let mut can_fit_examples: libc::c_int = ((*loader).B)
        .wrapping_div(4 as libc::c_int as libc::c_ulong) as libc::c_int;
    if can_fit_examples == 0 as libc::c_int {
        printf(
            b"HellaSwag EvalLoader: batch size %zu is < %d\n\0" as *const u8
                as *const libc::c_char,
            (*loader).B,
            4 as libc::c_int,
        );
        printf(
            b"---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b\n\0"
                as *const u8 as *const libc::c_char,
        );
        exit(1 as libc::c_int);
    }
    (*loader)
        .num_batches = (examples_per_process + can_fit_examples - 1 as libc::c_int)
        / can_fit_examples;
    (*loader).start_example_index = examples_per_process * (*loader).process_rank;
    (*loader)
        .end_example_index = examples_per_process
        * ((*loader).process_rank + 1 as libc::c_int);
    if (*loader).end_example_index > (*loader).num_examples {
        (*loader).end_example_index = (*loader).num_examples;
    }
    let mut header_bytes: int64_t = (256 as libc::c_int as libc::c_ulong)
        .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong) as int64_t;
    fseek_check(
        (*loader).eval_file,
        header_bytes as libc::c_int as libc::c_long,
        0 as libc::c_int,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        913 as libc::c_int,
    );
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < (*loader).start_example_index {
        let mut example_header: [uint16_t; 3] = [0; 3];
        fread_check(
            &mut *example_header.as_mut_ptr().offset(0 as libc::c_int as isize)
                as *mut uint16_t as *mut libc::c_void,
            ::core::mem::size_of::<uint16_t>() as libc::c_ulong,
            3 as libc::c_int as size_t,
            (*loader).eval_file,
            b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
            917 as libc::c_int,
        );
        let mut remaining_bytes: size_t = (example_header[1 as libc::c_int as usize]
            as libc::c_ulong)
            .wrapping_sub(
                (::core::mem::size_of::<uint16_t>() as libc::c_ulong)
                    .wrapping_mul(3 as libc::c_int as libc::c_ulong),
            );
        fseek_check(
            (*loader).eval_file,
            remaining_bytes as libc::c_int as libc::c_long,
            1 as libc::c_int,
            b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
            925 as libc::c_int,
        );
        i += 1;
        i;
    }
    (*loader).current_example_index = (*loader).start_example_index;
}
#[no_mangle]
pub unsafe extern "C" fn evalloader_init(
    mut loader: *mut EvalLoader,
    mut filename: *const libc::c_char,
    mut B: size_t,
    mut T: size_t,
    mut process_rank: libc::c_int,
    mut num_processes: libc::c_int,
) {
    (*loader).process_rank = process_rank;
    (*loader).num_processes = num_processes;
    (*loader).B = B;
    (*loader).T = T;
    (*loader)
        .eval_file = fopen_check(
        filename,
        b"rb\0" as *const u8 as *const libc::c_char,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        943 as libc::c_int,
    );
    let mut header: [libc::c_int; 256] = [0; 256];
    fread_check(
        header.as_mut_ptr() as *mut libc::c_void,
        ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
        256 as libc::c_int as size_t,
        (*loader).eval_file,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        946 as libc::c_int,
    );
    if header[0 as libc::c_int as usize] != 20240522 as libc::c_int {
        printf(b"Bad magic in eval file\n\0" as *const u8 as *const libc::c_char);
        exit(1 as libc::c_int);
    }
    if header[1 as libc::c_int as usize] != 1 as libc::c_int {
        printf(b"Bad version in data file\n\0" as *const u8 as *const libc::c_char);
        exit(1 as libc::c_int);
    }
    (*loader).num_examples = header[2 as libc::c_int as usize];
    let mut longest_example_bytes: size_t = header[3 as libc::c_int as usize] as size_t;
    let mut can_fit_examples: libc::c_int = B
        .wrapping_div(4 as libc::c_int as libc::c_ulong) as libc::c_int;
    (*loader)
        .buffer = malloc_check(
        longest_example_bytes,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        961 as libc::c_int,
    ) as *mut uint16_t;
    (*loader)
        .inputs = calloc(
        B.wrapping_mul(T),
        ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
    ) as *mut libc::c_int;
    (*loader)
        .targets = calloc(
        B.wrapping_mul(T),
        ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
    ) as *mut libc::c_int;
    (*loader)
        .mask = malloc_check(
        B
            .wrapping_mul(T)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        964 as libc::c_int,
    ) as *mut libc::c_char;
    (*loader)
        .label = malloc_check(
        (can_fit_examples as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        965 as libc::c_int,
    ) as *mut libc::c_int;
    evalloader_reset(loader);
}
#[no_mangle]
pub unsafe extern "C" fn evalloader_next_example_(
    mut loader: *mut EvalLoader,
    mut example_batch_index: libc::c_int,
) {
    let mut B: size_t = (*loader).B;
    let mut T: size_t = (*loader).T;
    let mut batch_dim_offset: libc::c_int = example_batch_index * 4 as libc::c_int;
    let mut example_header: [uint16_t; 3] = [0; 3];
    fread_check(
        &mut *example_header.as_mut_ptr().offset(0 as libc::c_int as isize)
            as *mut uint16_t as *mut libc::c_void,
        ::core::mem::size_of::<uint16_t>() as libc::c_ulong,
        3 as libc::c_int as size_t,
        (*loader).eval_file,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        981 as libc::c_int,
    );
    let mut example_bytes: size_t = (example_header[1 as libc::c_int as usize]
        as libc::c_ulong)
        .wrapping_sub(
            (::core::mem::size_of::<uint16_t>() as libc::c_ulong)
                .wrapping_mul(3 as libc::c_int as libc::c_ulong),
        );
    fread_check(
        (*loader).buffer as *mut libc::c_void,
        ::core::mem::size_of::<libc::c_char>() as libc::c_ulong,
        example_bytes,
        (*loader).eval_file,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        990 as libc::c_int,
    );
    let mut label: libc::c_int = *((*loader).buffer).offset(0 as libc::c_int as isize)
        as libc::c_int;
    let mut can_fit_examples: libc::c_int = ((*loader).B)
        .wrapping_div(4 as libc::c_int as libc::c_ulong) as libc::c_int;
    *((*loader).label).offset(example_batch_index as isize) = label;
    let mut num_completions: libc::c_int = *((*loader).buffer)
        .offset(1 as libc::c_int as isize) as libc::c_int;
    (*loader).num_completions = num_completions;
    let mut context_length: libc::c_int = *((*loader).buffer)
        .offset(2 as libc::c_int as isize) as libc::c_int;
    let mut context_tokens_start: *mut uint16_t = &mut *((*loader).buffer)
        .offset(3 as libc::c_int as isize) as *mut uint16_t;
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < num_completions {
        let mut i: libc::c_int = 0 as libc::c_int;
        while i < context_length {
            let mut boff: libc::c_int = batch_dim_offset + b;
            let mut tok_cur: libc::c_int = *context_tokens_start.offset(i as isize)
                as libc::c_int;
            *((*loader).inputs)
                .offset(
                    (boff as libc::c_ulong)
                        .wrapping_mul(T)
                        .wrapping_add(i as libc::c_ulong) as isize,
                ) = tok_cur;
            i += 1;
            i;
        }
        b += 1;
        b;
    }
    let mut completions_iter: *mut uint16_t = ((*loader).buffer)
        .offset(3 as libc::c_int as isize)
        .offset(context_length as isize);
    let mut c: libc::c_int = 0 as libc::c_int;
    while c < num_completions {
        let mut coff: libc::c_int = batch_dim_offset + c;
        let mut completion_length: libc::c_int = *completions_iter
            .offset(0 as libc::c_int as isize) as libc::c_int;
        let mut completion_tokens_start: *mut uint16_t = completions_iter
            .offset(1 as libc::c_int as isize);
        let mut i_0: libc::c_int = 0 as libc::c_int;
        while i_0 < completion_length {
            let mut tok_cur_0: libc::c_int = *completion_tokens_start
                .offset(i_0 as isize) as libc::c_int;
            *((*loader).inputs)
                .offset(
                    (coff as libc::c_ulong)
                        .wrapping_mul(T)
                        .wrapping_add(context_length as libc::c_ulong)
                        .wrapping_add(i_0 as libc::c_ulong) as isize,
                ) = tok_cur_0;
            *((*loader).targets)
                .offset(
                    (coff as libc::c_ulong)
                        .wrapping_mul(T)
                        .wrapping_add(context_length as libc::c_ulong)
                        .wrapping_add(i_0 as libc::c_ulong)
                        .wrapping_sub(1 as libc::c_int as libc::c_ulong) as isize,
                ) = tok_cur_0;
            *((*loader).mask)
                .offset(
                    (coff as libc::c_ulong)
                        .wrapping_mul(T)
                        .wrapping_add(context_length as libc::c_ulong)
                        .wrapping_add(i_0 as libc::c_ulong)
                        .wrapping_sub(1 as libc::c_int as libc::c_ulong) as isize,
                ) = 1 as libc::c_int as libc::c_char;
            i_0 += 1;
            i_0;
        }
        completions_iter = completions_iter
            .offset((1 as libc::c_int + completion_length) as isize);
        c += 1;
        c;
    }
    (*loader).current_example_index += 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn evalloader_next_batch(mut loader: *mut EvalLoader) {
    let mut B: size_t = (*loader).B;
    let mut T: size_t = (*loader).T;
    memset(
        (*loader).mask as *mut libc::c_void,
        0 as libc::c_int,
        B
            .wrapping_mul(T)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    );
    let mut can_fit_examples: libc::c_int = B
        .wrapping_div(4 as libc::c_int as libc::c_ulong) as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < can_fit_examples {
        if (*loader).current_example_index >= (*loader).end_example_index {
            break;
        }
        evalloader_next_example_(loader, i);
        i += 1;
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn evalloader_stat_losses(
    mut loader: *mut EvalLoader,
    mut losses: *mut f32,
) -> libc::c_int {
    let mut correct: libc::c_int = 0 as libc::c_int;
    let mut B: size_t = (*loader).B;
    let mut T: size_t = (*loader).T;
    let mut can_fit_examples: libc::c_int = B
        .wrapping_div(4 as libc::c_int as libc::c_ulong) as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < can_fit_examples {
        let mut min_loss: f32 = 0.0f32;
        let mut min_loss_index: libc::c_int = -(1 as libc::c_int);
        let mut active: libc::c_char = 0 as libc::c_int as libc::c_char;
        let mut b: libc::c_int = 0 as libc::c_int;
        while b < 4 as libc::c_int {
            let mut boff: libc::c_int = i * 4 as libc::c_int + b;
            let mut average_loss: f32 = 0.0f32;
            let mut count: libc::c_int = 0 as libc::c_int;
            let mut t: libc::c_int = 0 as libc::c_int;
            while (t as libc::c_ulong) < T {
                let mut mask: libc::c_char = *((*loader).mask)
                    .offset(
                        (boff as libc::c_ulong)
                            .wrapping_mul(T)
                            .wrapping_add(t as libc::c_ulong) as isize,
                    );
                if mask as libc::c_int == 1 as libc::c_int {
                    active = 1 as libc::c_int as libc::c_char;
                    average_loss
                        += *losses
                            .offset(
                                (boff as libc::c_ulong)
                                    .wrapping_mul(T)
                                    .wrapping_add(t as libc::c_ulong) as isize,
                            );
                    count += 1;
                    count;
                }
                t += 1;
                t;
            }
            if count > 0 as libc::c_int {
                average_loss /= count as f32;
            }
            if b == 0 as libc::c_int || average_loss < min_loss {
                min_loss = average_loss;
                min_loss_index = b;
            }
            b += 1;
            b;
        }
        if active as libc::c_int != 0
            && min_loss_index == *((*loader).label).offset(i as isize)
        {
            correct += 1 as libc::c_int;
        }
        i += 1;
        i;
    }
    return correct;
}
#[no_mangle]
pub unsafe extern "C" fn evalloader_free(mut loader: *mut EvalLoader) {
    free((*loader).buffer as *mut libc::c_void);
    free((*loader).inputs as *mut libc::c_void);
    free((*loader).targets as *mut libc::c_void);
    free((*loader).mask as *mut libc::c_void);
    free((*loader).label as *mut libc::c_void);
    fclose_check(
        (*loader).eval_file,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        1108 as libc::c_int,
    );
}
#[no_mangle]
pub static mut enzyme_dup: libc::c_int = 0;
#[no_mangle]
pub static mut enzyme_dupnoneed: libc::c_int = 0;
#[no_mangle]
pub static mut enzyme_const: libc::c_int = 0;
#[no_mangle]
pub static mut enzyme_width: libc::c_int = 0;
#[no_mangle]
pub unsafe extern "C" fn encoder_forward(
    mut out: *mut f32,
    mut inp: *mut libc::c_int,
    mut wte: *mut f32,
    mut wpe: *mut f32,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut C: libc::c_int,
) {
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut out_bt: *mut f32 = out
                .offset((b * T * C) as isize)
                .offset((t * C) as isize);
            let mut ix: libc::c_int = *inp.offset((b * T + t) as isize);
            let mut wte_ix: *mut f32 = wte.offset((ix * C) as isize);
            let mut wpe_t: *mut f32 = wpe.offset((t * C) as isize);
            let mut i: libc::c_int = 0 as libc::c_int;
            while i < C {
                *out_bt
                    .offset(
                        i as isize,
                    ) = *wte_ix.offset(i as isize) + *wpe_t.offset(i as isize);
                i += 1;
                i;
            }
            t += 1;
            t;
        }
        b += 1;
        b;
    }
}
#[no_mangle]
pub unsafe extern "C" fn layernorm_forward(
    mut out: *mut f32,
    mut mean: *mut f32,
    mut rstd: *mut f32,
    mut inp: *mut f32,
    mut weight: *mut f32,
    mut bias: *mut f32,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut C: libc::c_int,
) {
    let mut eps: f32 = 1e-5f32;
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut x: *mut f32 = inp
                .offset((b * T * C) as isize)
                .offset((t * C) as isize);
            let mut m: f32 = 0.0f32;
            let mut i: libc::c_int = 0 as libc::c_int;
            while i < C {
                m += *x.offset(i as isize);
                i += 1;
                i;
            }
            m = m / C as f32;
            let mut v: f32 = 0.0f32;
            let mut i_0: libc::c_int = 0 as libc::c_int;
            while i_0 < C {
                let mut xshift: f32 = *x.offset(i_0 as isize) - m;
                v += xshift * xshift;
                i_0 += 1;
                i_0;
            }
            v = v / C as f32;
            let mut s: f32 = 1.0f32 / sqrtf(v + eps);
            let mut out_bt: *mut f32 = out
                .offset((b * T * C) as isize)
                .offset((t * C) as isize);
            let mut i_1: libc::c_int = 0 as libc::c_int;
            while i_1 < C {
                let mut n: f32 = s * (*x.offset(i_1 as isize) - m);
                let mut o: f32 = n * *weight.offset(i_1 as isize)
                    + *bias.offset(i_1 as isize);
                *out_bt.offset(i_1 as isize) = o;
                i_1 += 1;
                i_1;
            }
            *mean.offset((b * T + t) as isize) = m;
            *rstd.offset((b * T + t) as isize) = s;
            t += 1;
            t;
        }
        b += 1;
        b;
    }
}
#[no_mangle]
pub unsafe extern "C" fn matmul_forward_naive(
    mut out: *mut f32,
    mut inp: *const f32,
    mut weight: *const f32,
    mut bias: *const f32,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut C: libc::c_int,
    mut OC: libc::c_int,
) {
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut bt: libc::c_int = b * T + t;
            let mut o: libc::c_int = 0 as libc::c_int;
            while o < OC {
                let mut val: f32 = if !bias.is_null() {
                    *bias.offset(o as isize)
                } else {
                    0.0f32
                };
                let mut i: libc::c_int = 0 as libc::c_int;
                while i < C {
                    val
                        += *inp.offset((bt * C + i) as isize)
                            * *weight.offset((o * C + i) as isize);
                    i += 1;
                    i;
                }
                *out.offset((bt * OC + o) as isize) = val;
                o += 1;
                o;
            }
            t += 1;
            t;
        }
        b += 1;
        b;
    }
}
#[no_mangle]
pub unsafe extern "C" fn matmul_forward(
    mut out: *mut f32,
    mut inp: *const f32,
    mut weight: *const f32,
    mut bias: *const f32,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut C: libc::c_int,
    mut OC: libc::c_int,
) {
    let LOOP_UNROLL: libc::c_int = 8 as libc::c_int;
    if B * T % LOOP_UNROLL != 0 as libc::c_int {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }
    let mut obt: libc::c_int = 0 as libc::c_int;
    while obt < B * T {
        let mut o: libc::c_int = 0 as libc::c_int;
        while o < OC {
            let vla = LOOP_UNROLL as usize;
            let mut result: Vec::<f32> = ::std::vec::from_elem(0., vla);
            let mut ibt: libc::c_int = 0 as libc::c_int;
            while ibt < LOOP_UNROLL {
                *result
                    .as_mut_ptr()
                    .offset(
                        ibt as isize,
                    ) = if !bias.is_null() { *bias.offset(o as isize) } else { 0.0f32 };
                ibt += 1;
                ibt;
            }
            let mut i: libc::c_int = 0 as libc::c_int;
            while i < C {
                let mut w: f32 = *weight.offset((i + o * C) as isize);
                let mut ibt_0: libc::c_int = 0 as libc::c_int;
                while ibt_0 < LOOP_UNROLL {
                    let mut bt: libc::c_int = obt + ibt_0;
                    *result.as_mut_ptr().offset(ibt_0 as isize)
                        += *inp.offset((bt * C + i) as isize) * w;
                    ibt_0 += 1;
                    ibt_0;
                }
                i += 1;
                i;
            }
            let mut ibt_1: libc::c_int = 0 as libc::c_int;
            while ibt_1 < LOOP_UNROLL {
                let mut bt_0: libc::c_int = obt + ibt_1;
                *out
                    .offset(
                        (bt_0 * OC + o) as isize,
                    ) = *result.as_mut_ptr().offset(ibt_1 as isize);
                ibt_1 += 1;
                ibt_1;
            }
            o += 1;
            o;
        }
        obt += LOOP_UNROLL;
    }
}
#[no_mangle]
pub unsafe extern "C" fn attention_forward(
    mut out: *mut f32,
    mut preatt: *mut f32,
    mut att: *mut f32,
    mut inp: *mut f32,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut C: libc::c_int,
    mut NH: libc::c_int,
) {
    let mut C3: libc::c_int = C * 3 as libc::c_int;
    let mut hs: libc::c_int = C / NH;
    let mut scale: f32 = (1.0f64
        / sqrtf(hs as f32) as libc::c_double) as f32;
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut h: libc::c_int = 0 as libc::c_int;
            while h < NH {
                let mut query_t: *mut f32 = inp
                    .offset((b * T * C3) as isize)
                    .offset((t * C3) as isize)
                    .offset((h * hs) as isize);
                let mut preatt_bth: *mut f32 = preatt
                    .offset((b * NH * T * T) as isize)
                    .offset((h * T * T) as isize)
                    .offset((t * T) as isize);
                let mut att_bth: *mut f32 = att
                    .offset((b * NH * T * T) as isize)
                    .offset((h * T * T) as isize)
                    .offset((t * T) as isize);
                let mut maxval: f32 = -10000.0f32;
                let mut t2: libc::c_int = 0 as libc::c_int;
                while t2 <= t {
                    let mut key_t2: *mut f32 = inp
                        .offset((b * T * C3) as isize)
                        .offset((t2 * C3) as isize)
                        .offset((h * hs) as isize)
                        .offset(C as isize);
                    let mut val: f32 = 0.0f32;
                    let mut i: libc::c_int = 0 as libc::c_int;
                    while i < hs {
                        val += *query_t.offset(i as isize) * *key_t2.offset(i as isize);
                        i += 1;
                        i;
                    }
                    val *= scale;
                    if val > maxval {
                        maxval = val;
                    }
                    *preatt_bth.offset(t2 as isize) = val;
                    t2 += 1;
                    t2;
                }
                let mut expsum: f32 = 0.0f32;
                let mut t2_0: libc::c_int = 0 as libc::c_int;
                while t2_0 <= t {
                    let mut expv: f32 = expf(
                        *preatt_bth.offset(t2_0 as isize) - maxval,
                    );
                    expsum += expv;
                    *att_bth.offset(t2_0 as isize) = expv;
                    t2_0 += 1;
                    t2_0;
                }
                let mut expsum_inv: f32 = if expsum == 0.0f32 {
                    0.0f32
                } else {
                    1.0f32 / expsum
                };
                let mut t2_1: libc::c_int = 0 as libc::c_int;
                while t2_1 < T {
                    if t2_1 <= t {
                        *att_bth.offset(t2_1 as isize) *= expsum_inv;
                    } else {
                        *att_bth.offset(t2_1 as isize) = 0.0f32;
                    }
                    t2_1 += 1;
                    t2_1;
                }
                let mut out_bth: *mut f32 = out
                    .offset((b * T * C) as isize)
                    .offset((t * C) as isize)
                    .offset((h * hs) as isize);
                let mut i_0: libc::c_int = 0 as libc::c_int;
                while i_0 < hs {
                    *out_bth.offset(i_0 as isize) = 0.0f32;
                    i_0 += 1;
                    i_0;
                }
                let mut t2_2: libc::c_int = 0 as libc::c_int;
                while t2_2 <= t {
                    let mut value_t2: *mut f32 = inp
                        .offset((b * T * C3) as isize)
                        .offset((t2_2 * C3) as isize)
                        .offset((h * hs) as isize)
                        .offset((C * 2 as libc::c_int) as isize);
                    let mut att_btht2: f32 = *att_bth.offset(t2_2 as isize);
                    let mut i_1: libc::c_int = 0 as libc::c_int;
                    while i_1 < hs {
                        *out_bth.offset(i_1 as isize)
                            += att_btht2 * *value_t2.offset(i_1 as isize);
                        i_1 += 1;
                        i_1;
                    }
                    t2_2 += 1;
                    t2_2;
                }
                h += 1;
                h;
            }
            t += 1;
            t;
        }
        b += 1;
        b;
    }
}
#[no_mangle]
pub unsafe extern "C" fn gelu_forward(
    mut out: *mut f32,
    mut inp: *mut f32,
    mut N: libc::c_int,
) {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < N {
        let mut x: f32 = *inp.offset(i as isize);
        let mut cube: f32 = 0.044715f32 * x * x * x;
        *out
            .offset(
                i as isize,
            ) = 0.5f32 * x
            * (1.0f32
                + tanhf(
                    sqrtf(
                        (2.0f32 as libc::c_double / 3.14159265358979323846f64)
                            as f32,
                    ) * (x + cube),
                ));
        i += 1;
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn residual_forward(
    mut out: *mut f32,
    mut inp1: *mut f32,
    mut inp2: *mut f32,
    mut N: libc::c_int,
) {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < N {
        *out.offset(i as isize) = *inp1.offset(i as isize) + *inp2.offset(i as isize);
        i += 1;
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn softmax_forward(
    mut probs: *mut f32,
    mut logits: *mut f32,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut V: libc::c_int,
    mut Vp: libc::c_int,
) {
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut logits_bt: *mut f32 = logits
                .offset((b * T * Vp) as isize)
                .offset((t * Vp) as isize);
            let mut probs_bt: *mut f32 = probs
                .offset((b * T * Vp) as isize)
                .offset((t * Vp) as isize);
            let mut maxval: f32 = -10000.0f32;
            let mut i: libc::c_int = 0 as libc::c_int;
            while i < V {
                if *logits_bt.offset(i as isize) > maxval {
                    maxval = *logits_bt.offset(i as isize);
                }
                i += 1;
                i;
            }
            let mut sum: f32 = 0.0f32;
            let mut i_0: libc::c_int = 0 as libc::c_int;
            while i_0 < V {
                *probs_bt
                    .offset(
                        i_0 as isize,
                    ) = expf(*logits_bt.offset(i_0 as isize) - maxval);
                sum += *probs_bt.offset(i_0 as isize);
                i_0 += 1;
                i_0;
            }
            let mut i_1: libc::c_int = 0 as libc::c_int;
            while i_1 < V {
                *probs_bt.offset(i_1 as isize) /= sum;
                i_1 += 1;
                i_1;
            }
            let mut i_2: libc::c_int = V;
            while i_2 < Vp {
                *probs_bt.offset(i_2 as isize) = 0.0f32;
                i_2 += 1;
                i_2;
            }
            t += 1;
            t;
        }
        b += 1;
        b;
    }
}
#[no_mangle]
pub unsafe extern "C" fn crossentropy_forward(
    mut losses: *mut f32,
    mut probs: *mut f32,
    mut targets: *mut libc::c_int,
    mut B: libc::c_int,
    mut T: libc::c_int,
    mut Vp: libc::c_int,
) {
    let mut b: libc::c_int = 0 as libc::c_int;
    while b < B {
        let mut t: libc::c_int = 0 as libc::c_int;
        while t < T {
            let mut probs_bt: *mut f32 = probs
                .offset((b * T * Vp) as isize)
                .offset((t * Vp) as isize);
            let mut ix: libc::c_int = *targets.offset((b * T + t) as isize);
            *losses.offset((b * T + t) as isize) = -logf(*probs_bt.offset(ix as isize));
            t += 1;
            t;
        }
        b += 1;
        b;
    }
}
#[no_mangle]
pub unsafe extern "C" fn fill_in_parameter_sizes(
    mut param_sizes: *mut size_t,
    mut config: GPT2Config,
) {
    let mut Vp: size_t = config.padded_vocab_size as size_t;
    let mut C: size_t = config.channels as size_t;
    let mut maxT: size_t = config.max_seq_len as size_t;
    let mut L: size_t = config.num_layers as size_t;
    *param_sizes.offset(0 as libc::c_int as isize) = Vp.wrapping_mul(C);
    *param_sizes.offset(1 as libc::c_int as isize) = maxT.wrapping_mul(C);
    *param_sizes.offset(2 as libc::c_int as isize) = L.wrapping_mul(C);
    *param_sizes.offset(3 as libc::c_int as isize) = L.wrapping_mul(C);
    *param_sizes
        .offset(
            4 as libc::c_int as isize,
        ) = L
        .wrapping_mul((3 as libc::c_int as libc::c_ulong).wrapping_mul(C))
        .wrapping_mul(C);
    *param_sizes
        .offset(
            5 as libc::c_int as isize,
        ) = L.wrapping_mul((3 as libc::c_int as libc::c_ulong).wrapping_mul(C));
    *param_sizes.offset(6 as libc::c_int as isize) = L.wrapping_mul(C).wrapping_mul(C);
    *param_sizes.offset(7 as libc::c_int as isize) = L.wrapping_mul(C);
    *param_sizes.offset(8 as libc::c_int as isize) = L.wrapping_mul(C);
    *param_sizes.offset(9 as libc::c_int as isize) = L.wrapping_mul(C);
    *param_sizes
        .offset(
            10 as libc::c_int as isize,
        ) = L
        .wrapping_mul((4 as libc::c_int as libc::c_ulong).wrapping_mul(C))
        .wrapping_mul(C);
    *param_sizes
        .offset(
            11 as libc::c_int as isize,
        ) = L.wrapping_mul((4 as libc::c_int as libc::c_ulong).wrapping_mul(C));
    *param_sizes
        .offset(
            12 as libc::c_int as isize,
        ) = L
        .wrapping_mul(C)
        .wrapping_mul((4 as libc::c_int as libc::c_ulong).wrapping_mul(C));
    *param_sizes.offset(13 as libc::c_int as isize) = L.wrapping_mul(C);
    *param_sizes.offset(14 as libc::c_int as isize) = C;
    *param_sizes.offset(15 as libc::c_int as isize) = C;
}
#[no_mangle]
pub unsafe extern "C" fn malloc_and_point_parameters(
    mut params: *mut ParameterTensors,
    mut param_sizes: *mut size_t,
) -> *mut f32 {
    let mut num_parameters: size_t = 0 as libc::c_int as size_t;
    let mut i: size_t = 0 as libc::c_int as size_t;
    while i < 16 as libc::c_int as libc::c_ulong {
        num_parameters = (num_parameters as libc::c_ulong)
            .wrapping_add(*param_sizes.offset(i as isize)) as size_t as size_t;
        i = i.wrapping_add(1);
        i;
    }
    let mut params_memory: *mut f32 = malloc_check(
        num_parameters
            .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        1466 as libc::c_int,
    ) as *mut f32;
    let mut ptrs: [*mut *mut f32; 16] = [
        &mut (*params).wte,
        &mut (*params).wpe,
        &mut (*params).ln1w,
        &mut (*params).ln1b,
        &mut (*params).qkvw,
        &mut (*params).qkvb,
        &mut (*params).attprojw,
        &mut (*params).attprojb,
        &mut (*params).ln2w,
        &mut (*params).ln2b,
        &mut (*params).fcw,
        &mut (*params).fcb,
        &mut (*params).fcprojw,
        &mut (*params).fcprojb,
        &mut (*params).lnfw,
        &mut (*params).lnfb,
    ];
    let mut params_memory_iterator: *mut f32 = params_memory;
    let mut i_0: size_t = 0 as libc::c_int as size_t;
    while i_0 < 16 as libc::c_int as libc::c_ulong {
        *ptrs[i_0 as usize] = params_memory_iterator;
        params_memory_iterator = params_memory_iterator
            .offset(*param_sizes.offset(i_0 as isize) as isize);
        i_0 = i_0.wrapping_add(1);
        i_0;
    }
    return params_memory;
}
#[no_mangle]
pub unsafe extern "C" fn fill_in_activation_sizes(
    mut act_sizes: *mut size_t,
    mut config: GPT2Config,
    mut B: libc::c_int,
    mut T: libc::c_int,
) {
    let mut C: size_t = config.channels as size_t;
    let mut NH: size_t = config.num_heads as size_t;
    let mut L: size_t = config.num_layers as size_t;
    let mut Vp: size_t = config.padded_vocab_size as size_t;
    *act_sizes
        .offset(0 as libc::c_int as isize) = ((B * T) as libc::c_ulong).wrapping_mul(C);
    *act_sizes
        .offset(
            1 as libc::c_int as isize,
        ) = L
        .wrapping_mul(B as libc::c_ulong)
        .wrapping_mul(T as libc::c_ulong)
        .wrapping_mul(C);
    *act_sizes
        .offset(
            2 as libc::c_int as isize,
        ) = L.wrapping_mul(B as libc::c_ulong).wrapping_mul(T as libc::c_ulong);
    *act_sizes
        .offset(
            3 as libc::c_int as isize,
        ) = L.wrapping_mul(B as libc::c_ulong).wrapping_mul(T as libc::c_ulong);
    *act_sizes
        .offset(
            4 as libc::c_int as isize,
        ) = L
        .wrapping_mul(B as libc::c_ulong)
        .wrapping_mul(T as libc::c_ulong)
        .wrapping_mul(3 as libc::c_int as libc::c_ulong)
        .wrapping_mul(C);
    *act_sizes
        .offset(
            5 as libc::c_int as isize,
        ) = L
        .wrapping_mul(B as libc::c_ulong)
        .wrapping_mul(T as libc::c_ulong)
        .wrapping_mul(C);
    *act_sizes
        .offset(
            6 as libc::c_int as isize,
        ) = L
        .wrapping_mul(B as libc::c_ulong)
        .wrapping_mul(NH)
        .wrapping_mul(T as libc::c_ulong)
        .wrapping_mul(T as libc::c_ulong);
    *act_sizes
        .offset(
            7 as libc::c_int as isize,
        ) = L
        .wrapping_mul(B as libc::c_ulong)
        .wrapping_mul(NH)
        .wrapping_mul(T as libc::c_ulong)
        .wrapping_mul(T as libc::c_ulong);
    *act_sizes
        .offset(
            8 as libc::c_int as isize,
        ) = L
        .wrapping_mul(B as libc::c_ulong)
        .wrapping_mul(T as libc::c_ulong)
        .wrapping_mul(C);
    *act_sizes
        .offset(
            9 as libc::c_int as isize,
        ) = L
        .wrapping_mul(B as libc::c_ulong)
        .wrapping_mul(T as libc::c_ulong)
        .wrapping_mul(C);
    *act_sizes
        .offset(
            10 as libc::c_int as isize,
        ) = L
        .wrapping_mul(B as libc::c_ulong)
        .wrapping_mul(T as libc::c_ulong)
        .wrapping_mul(C);
    *act_sizes
        .offset(
            11 as libc::c_int as isize,
        ) = L.wrapping_mul(B as libc::c_ulong).wrapping_mul(T as libc::c_ulong);
    *act_sizes
        .offset(
            12 as libc::c_int as isize,
        ) = L.wrapping_mul(B as libc::c_ulong).wrapping_mul(T as libc::c_ulong);
    *act_sizes
        .offset(
            13 as libc::c_int as isize,
        ) = L
        .wrapping_mul(B as libc::c_ulong)
        .wrapping_mul(T as libc::c_ulong)
        .wrapping_mul(4 as libc::c_int as libc::c_ulong)
        .wrapping_mul(C);
    *act_sizes
        .offset(
            14 as libc::c_int as isize,
        ) = L
        .wrapping_mul(B as libc::c_ulong)
        .wrapping_mul(T as libc::c_ulong)
        .wrapping_mul(4 as libc::c_int as libc::c_ulong)
        .wrapping_mul(C);
    *act_sizes
        .offset(
            15 as libc::c_int as isize,
        ) = L
        .wrapping_mul(B as libc::c_ulong)
        .wrapping_mul(T as libc::c_ulong)
        .wrapping_mul(C);
    *act_sizes
        .offset(
            16 as libc::c_int as isize,
        ) = L
        .wrapping_mul(B as libc::c_ulong)
        .wrapping_mul(T as libc::c_ulong)
        .wrapping_mul(C);
    *act_sizes
        .offset(17 as libc::c_int as isize) = ((B * T) as libc::c_ulong).wrapping_mul(C);
    *act_sizes.offset(18 as libc::c_int as isize) = (B * T) as size_t;
    *act_sizes.offset(19 as libc::c_int as isize) = (B * T) as size_t;
    *act_sizes
        .offset(
            20 as libc::c_int as isize,
        ) = ((B * T) as libc::c_ulong).wrapping_mul(Vp);
    *act_sizes
        .offset(
            21 as libc::c_int as isize,
        ) = ((B * T) as libc::c_ulong).wrapping_mul(Vp);
    *act_sizes.offset(22 as libc::c_int as isize) = (B * T) as size_t;
}
#[no_mangle]
pub unsafe extern "C" fn malloc_and_point_activations(
    mut acts: *mut ActivationTensors,
    mut act_sizes: *mut size_t,
) -> *mut f32 {
    let mut num_activations: size_t = 0 as libc::c_int as size_t;
    let mut i: size_t = 0 as libc::c_int as size_t;
    while i < 23 as libc::c_int as libc::c_ulong {
        num_activations = (num_activations as libc::c_ulong)
            .wrapping_add(*act_sizes.offset(i as isize)) as size_t as size_t;
        i = i.wrapping_add(1);
        i;
    }
    let mut acts_memory: *mut f32 = malloc_check(
        num_activations
            .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        1543 as libc::c_int,
    ) as *mut f32;
    let mut ptrs: [*mut *mut f32; 23] = [
        &mut (*acts).encoded,
        &mut (*acts).ln1,
        &mut (*acts).ln1_mean,
        &mut (*acts).ln1_rstd,
        &mut (*acts).qkv,
        &mut (*acts).atty,
        &mut (*acts).preatt,
        &mut (*acts).att,
        &mut (*acts).attproj,
        &mut (*acts).residual2,
        &mut (*acts).ln2,
        &mut (*acts).ln2_mean,
        &mut (*acts).ln2_rstd,
        &mut (*acts).fch,
        &mut (*acts).fch_gelu,
        &mut (*acts).fcproj,
        &mut (*acts).residual3,
        &mut (*acts).lnf,
        &mut (*acts).lnf_mean,
        &mut (*acts).lnf_rstd,
        &mut (*acts).logits,
        &mut (*acts).probs,
        &mut (*acts).losses,
    ];
    let mut acts_memory_iterator: *mut f32 = acts_memory;
    let mut i_0: size_t = 0 as libc::c_int as size_t;
    while i_0 < 23 as libc::c_int as libc::c_ulong {
        *ptrs[i_0 as usize] = acts_memory_iterator;
        acts_memory_iterator = acts_memory_iterator
            .offset(*act_sizes.offset(i_0 as isize) as isize);
        i_0 = i_0.wrapping_add(1);
        i_0;
    }
    return acts_memory;
}
#[no_mangle]
pub unsafe extern "C" fn zero_parameters(
    mut params: *mut ParameterTensors,
    mut param_sizes: *mut size_t,
) {
    if !((*params).wte).is_null() {
        memset(
            (*params).wte as *mut libc::c_void,
            0 as libc::c_int,
            (*param_sizes.offset(0 as libc::c_int as isize))
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
    if !((*params).wpe).is_null() {
        memset(
            (*params).wpe as *mut libc::c_void,
            0 as libc::c_int,
            (*param_sizes.offset(1 as libc::c_int as isize))
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
    if !((*params).ln1w).is_null() {
        memset(
            (*params).ln1w as *mut libc::c_void,
            0 as libc::c_int,
            (*param_sizes.offset(2 as libc::c_int as isize))
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
    if !((*params).ln1b).is_null() {
        memset(
            (*params).ln1b as *mut libc::c_void,
            0 as libc::c_int,
            (*param_sizes.offset(3 as libc::c_int as isize))
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
    if !((*params).qkvw).is_null() {
        memset(
            (*params).qkvw as *mut libc::c_void,
            0 as libc::c_int,
            (*param_sizes.offset(4 as libc::c_int as isize))
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
    if !((*params).qkvb).is_null() {
        memset(
            (*params).qkvb as *mut libc::c_void,
            0 as libc::c_int,
            (*param_sizes.offset(5 as libc::c_int as isize))
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
    if !((*params).attprojw).is_null() {
        memset(
            (*params).attprojw as *mut libc::c_void,
            0 as libc::c_int,
            (*param_sizes.offset(6 as libc::c_int as isize))
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
    if !((*params).attprojb).is_null() {
        memset(
            (*params).attprojb as *mut libc::c_void,
            0 as libc::c_int,
            (*param_sizes.offset(7 as libc::c_int as isize))
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
    if !((*params).ln2w).is_null() {
        memset(
            (*params).ln2w as *mut libc::c_void,
            0 as libc::c_int,
            (*param_sizes.offset(8 as libc::c_int as isize))
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
    if !((*params).ln2b).is_null() {
        memset(
            (*params).ln2b as *mut libc::c_void,
            0 as libc::c_int,
            (*param_sizes.offset(9 as libc::c_int as isize))
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
    if !((*params).fcw).is_null() {
        memset(
            (*params).fcw as *mut libc::c_void,
            0 as libc::c_int,
            (*param_sizes.offset(10 as libc::c_int as isize))
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
    if !((*params).fcb).is_null() {
        memset(
            (*params).fcb as *mut libc::c_void,
            0 as libc::c_int,
            (*param_sizes.offset(11 as libc::c_int as isize))
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
    if !((*params).fcprojw).is_null() {
        memset(
            (*params).fcprojw as *mut libc::c_void,
            0 as libc::c_int,
            (*param_sizes.offset(12 as libc::c_int as isize))
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
    if !((*params).fcprojb).is_null() {
        memset(
            (*params).fcprojb as *mut libc::c_void,
            0 as libc::c_int,
            (*param_sizes.offset(13 as libc::c_int as isize))
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
    if !((*params).lnfw).is_null() {
        memset(
            (*params).lnfw as *mut libc::c_void,
            0 as libc::c_int,
            (*param_sizes.offset(14 as libc::c_int as isize))
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
    if !((*params).lnfb).is_null() {
        memset(
            (*params).lnfb as *mut libc::c_void,
            0 as libc::c_int,
            (*param_sizes.offset(15 as libc::c_int as isize))
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
}
#[no_mangle]
pub unsafe extern "C" fn gpt2_build_from_checkpoint(
    mut model: *mut GPT2,
    mut const_model: *mut GPT2Const,
    mut checkpoint_path: *const libc::c_char,
) {
    let mut model_file: *mut FILE = fopen_check(
        checkpoint_path,
        b"rb\0" as *const u8 as *const libc::c_char,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        1645 as libc::c_int,
    );
    let mut model_header: [libc::c_int; 256] = [0; 256];
    fread_check(
        model_header.as_mut_ptr() as *mut libc::c_void,
        ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
        256 as libc::c_int as size_t,
        model_file,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        1647 as libc::c_int,
    );
    if model_header[0 as libc::c_int as usize] != 20240326 as libc::c_int {
        printf(b"Bad magic model file\n\0" as *const u8 as *const libc::c_char);
        exit(1 as libc::c_int);
    }
    if model_header[1 as libc::c_int as usize] != 3 as libc::c_int {
        printf(b"Bad version in model file\n\0" as *const u8 as *const libc::c_char);
        printf(
            b"---> HINT: try to re-run `python train_gpt2.py`\n\0" as *const u8
                as *const libc::c_char,
        );
        exit(1 as libc::c_int);
    }
    let mut maxT: size_t = 0;
    let mut V: size_t = 0;
    let mut Vp: size_t = 0;
    let mut L: size_t = 0;
    let mut NH: size_t = 0;
    let mut C: size_t = 0;
    maxT = model_header[2 as libc::c_int as usize] as size_t;
    (*const_model).config.max_seq_len = maxT as libc::c_int;
    V = model_header[3 as libc::c_int as usize] as size_t;
    (*const_model).config.vocab_size = V as libc::c_int;
    L = model_header[4 as libc::c_int as usize] as size_t;
    (*const_model).config.num_layers = L as libc::c_int;
    NH = model_header[5 as libc::c_int as usize] as size_t;
    (*const_model).config.num_heads = NH as libc::c_int;
    C = model_header[6 as libc::c_int as usize] as size_t;
    (*const_model).config.channels = C as libc::c_int;
    Vp = model_header[7 as libc::c_int as usize] as size_t;
    (*const_model).config.padded_vocab_size = Vp as libc::c_int;
    printf(b"[GPT-2]\n\0" as *const u8 as *const libc::c_char);
    printf(b"max_seq_len: %zu\n\0" as *const u8 as *const libc::c_char, maxT);
    printf(b"vocab_size: %zu\n\0" as *const u8 as *const libc::c_char, V);
    printf(b"padded_vocab_size: %zu\n\0" as *const u8 as *const libc::c_char, Vp);
    printf(b"num_layers: %zu\n\0" as *const u8 as *const libc::c_char, L);
    printf(b"num_heads: %zu\n\0" as *const u8 as *const libc::c_char, NH);
    printf(b"channels: %zu\n\0" as *const u8 as *const libc::c_char, C);
    fill_in_parameter_sizes(((*model).param_sizes).as_mut_ptr(), (*const_model).config);
    let mut num_parameters: size_t = 0 as libc::c_int as size_t;
    let mut i: size_t = 0 as libc::c_int as size_t;
    while i < 16 as libc::c_int as libc::c_ulong {
        num_parameters = (num_parameters as libc::c_ulong)
            .wrapping_add((*model).param_sizes[i as usize]) as size_t as size_t;
        i = i.wrapping_add(1);
        i;
    }
    printf(
        b"num_parameters: %zu\n\0" as *const u8 as *const libc::c_char,
        num_parameters,
    );
    (*const_model).num_parameters = num_parameters;
    (*model)
        .params_memory = malloc_and_point_parameters(
        &mut (*model).params,
        ((*model).param_sizes).as_mut_ptr(),
    );
    fread_check(
        (*model).params_memory as *mut libc::c_void,
        ::core::mem::size_of::<f32>() as libc::c_ulong,
        num_parameters,
        model_file,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        1684 as libc::c_int,
    );
    fclose_check(
        model_file,
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        1685 as libc::c_int,
    );
    (*model).acts_memory = 0 as *mut f32;
    (*const_model).grads_memory = 0 as *mut f32;
    (*const_model).m_memory = 0 as *mut f32;
    (*const_model).v_memory = 0 as *mut f32;
    (*const_model).grads_acts_memory = 0 as *mut f32;
    (*const_model).inputs = 0 as *mut libc::c_int;
    (*const_model).targets = 0 as *mut libc::c_int;
    (*const_model).batch_size = 0 as libc::c_int;
    (*const_model).seq_len = 0 as libc::c_int;
    (*const_model).mean_loss = -1.0f32;
}
#[no_mangle]
pub unsafe extern "C" fn gpt2_init(
    mut model: *mut GPT2,
    mut model_const: *mut GPT2Const,
    mut B: size_t,
    mut T: size_t,
) {
    if ((*model).params_memory).is_null() {
        printf(
            b"Error: model was not initialized properly.\n\0" as *const u8
                as *const libc::c_char,
        );
        exit(1 as libc::c_int);
    }
    let mut V: size_t = (*model_const).config.vocab_size as size_t;
    let mut Vp: size_t = (*model_const).config.padded_vocab_size as size_t;
    if ((*model).acts_memory).is_null() {
        (*model_const).batch_size = B as libc::c_int;
        (*model_const).seq_len = T as libc::c_int;
        fill_in_activation_sizes(
            ((*model).act_sizes).as_mut_ptr(),
            (*model_const).config,
            B as libc::c_int,
            T as libc::c_int,
        );
        let mut num_activations: size_t = 0 as libc::c_int as size_t;
        let mut i: size_t = 0 as libc::c_int as size_t;
        while i < 23 as libc::c_int as libc::c_ulong {
            num_activations = (num_activations as libc::c_ulong)
                .wrapping_add((*model).act_sizes[i as usize]) as size_t as size_t;
            i = i.wrapping_add(1);
            i;
        }
        printf(
            b"num_activations: %zu\n\0" as *const u8 as *const libc::c_char,
            num_activations,
        );
        (*model_const).num_activations = num_activations;
        (*model)
            .acts_memory = malloc_and_point_activations(
            &mut (*model).acts,
            ((*model).act_sizes).as_mut_ptr(),
        );
        (*model_const)
            .inputs = malloc_check(
            B
                .wrapping_mul(T)
                .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
            b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
            1727 as libc::c_int,
        ) as *mut libc::c_int;
        (*model_const)
            .targets = malloc_check(
            B
                .wrapping_mul(T)
                .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
            b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
            1728 as libc::c_int,
        ) as *mut libc::c_int;
    } else if B != (*model_const).batch_size as libc::c_ulong
        || T != (*model_const).seq_len as libc::c_ulong
    {
        printf(
            b"Model: B=%d T=%d, Desired: B=%d T=%d\n\0" as *const u8
                as *const libc::c_char,
            (*model_const).batch_size,
            (*model_const).seq_len,
            B as libc::c_int,
            T as libc::c_int,
        );
        exit(1 as libc::c_int);
    }
}
#[no_mangle]
#[autodiff(gpt2_ad_backward, Reverse, Duplicated, Const, Const, Const, Const, Const, Duplicated)]
pub unsafe extern "C" fn gpt2_forward(
    mut model: *mut GPT2,
    mut model_consts: *mut GPT2Const,
    mut inputs: *mut libc::c_int,
    mut targets: *mut libc::c_int,
    mut B: size_t,
    mut T: size_t,
    mut res: *mut f32,
) {
    let mut V: size_t = (*model_consts).config.vocab_size as size_t;
    let mut Vp: size_t = (*model_consts).config.padded_vocab_size as size_t;
    let mut i: libc::c_int = 0 as libc::c_int;
    while (i as libc::c_ulong) < B.wrapping_mul(T) {
        !targets.is_null();
        i += 1;
        i;
    }
    let mut params: ParameterTensors = (*model).params;
    let mut acts: ActivationTensors = (*model).acts;
    let mut residual: *mut f32 = 0 as *mut f32;
    encoder_forward(
        acts.encoded,
        inputs,
        params.wte,
        params.wpe,
        B as libc::c_int,
        T as libc::c_int,
        (*model_consts).config.channels,
    );
    let mut l: libc::c_int = 0 as libc::c_int;
    while l < (*model_consts).config.num_layers {
        let mut C: size_t = (*model_consts).config.channels as size_t;
        let mut NH: size_t = (*model_consts).config.num_heads as size_t;
        residual = if l == 0 as libc::c_int {
            acts.encoded
        } else {
            (acts.residual3)
                .offset(
                    ((l - 1 as libc::c_int) as libc::c_ulong)
                        .wrapping_mul(B)
                        .wrapping_mul(T)
                        .wrapping_mul(C) as isize,
                )
        };
        let mut l_ln1w: *mut f32 = (params.ln1w)
            .offset((l as libc::c_ulong).wrapping_mul(C) as isize);
        let mut l_ln1b: *mut f32 = (params.ln1b)
            .offset((l as libc::c_ulong).wrapping_mul(C) as isize);
        let mut l_qkvw: *mut f32 = (params.qkvw)
            .offset(
                ((l * 3 as libc::c_int) as libc::c_ulong).wrapping_mul(C).wrapping_mul(C)
                    as isize,
            );
        let mut l_qkvb: *mut f32 = (params.qkvb)
            .offset(((l * 3 as libc::c_int) as libc::c_ulong).wrapping_mul(C) as isize);
        let mut l_attprojw: *mut f32 = (params.attprojw)
            .offset((l as libc::c_ulong).wrapping_mul(C).wrapping_mul(C) as isize);
        let mut l_attprojb: *mut f32 = (params.attprojb)
            .offset((l as libc::c_ulong).wrapping_mul(C) as isize);
        let mut l_ln2w: *mut f32 = (params.ln2w)
            .offset((l as libc::c_ulong).wrapping_mul(C) as isize);
        let mut l_ln2b: *mut f32 = (params.ln2b)
            .offset((l as libc::c_ulong).wrapping_mul(C) as isize);
        let mut l_fcw: *mut f32 = (params.fcw)
            .offset(
                ((l * 4 as libc::c_int) as libc::c_ulong).wrapping_mul(C).wrapping_mul(C)
                    as isize,
            );
        let mut l_fcb: *mut f32 = (params.fcb)
            .offset(((l * 4 as libc::c_int) as libc::c_ulong).wrapping_mul(C) as isize);
        let mut l_fcprojw: *mut f32 = (params.fcprojw)
            .offset(
                (l as libc::c_ulong)
                    .wrapping_mul(C)
                    .wrapping_mul(4 as libc::c_int as libc::c_ulong)
                    .wrapping_mul(C) as isize,
            );
        let mut l_fcprojb: *mut f32 = (params.fcprojb)
            .offset((l as libc::c_ulong).wrapping_mul(C) as isize);
        let mut l_ln1: *mut f32 = (acts.ln1)
            .offset(
                (l as libc::c_ulong).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C)
                    as isize,
            );
        let mut l_ln1_mean: *mut f32 = (acts.ln1_mean)
            .offset((l as libc::c_ulong).wrapping_mul(B).wrapping_mul(T) as isize);
        let mut l_ln1_rstd: *mut f32 = (acts.ln1_rstd)
            .offset((l as libc::c_ulong).wrapping_mul(B).wrapping_mul(T) as isize);
        let mut l_qkv: *mut f32 = (acts.qkv)
            .offset(
                (l as libc::c_ulong)
                    .wrapping_mul(B)
                    .wrapping_mul(T)
                    .wrapping_mul(3 as libc::c_int as libc::c_ulong)
                    .wrapping_mul(C) as isize,
            );
        let mut l_atty: *mut f32 = (acts.atty)
            .offset(
                (l as libc::c_ulong).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C)
                    as isize,
            );
        let mut l_preatt: *mut f32 = (acts.preatt)
            .offset(
                (l as libc::c_ulong)
                    .wrapping_mul(B)
                    .wrapping_mul(NH)
                    .wrapping_mul(T)
                    .wrapping_mul(T) as isize,
            );
        let mut l_att: *mut f32 = (acts.att)
            .offset(
                (l as libc::c_ulong)
                    .wrapping_mul(B)
                    .wrapping_mul(NH)
                    .wrapping_mul(T)
                    .wrapping_mul(T) as isize,
            );
        let mut l_attproj: *mut f32 = (acts.attproj)
            .offset(
                (l as libc::c_ulong).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C)
                    as isize,
            );
        let mut l_residual2: *mut f32 = (acts.residual2)
            .offset(
                (l as libc::c_ulong).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C)
                    as isize,
            );
        let mut l_ln2: *mut f32 = (acts.ln2)
            .offset(
                (l as libc::c_ulong).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C)
                    as isize,
            );
        let mut l_ln2_mean: *mut f32 = (acts.ln2_mean)
            .offset((l as libc::c_ulong).wrapping_mul(B).wrapping_mul(T) as isize);
        let mut l_ln2_rstd: *mut f32 = (acts.ln2_rstd)
            .offset((l as libc::c_ulong).wrapping_mul(B).wrapping_mul(T) as isize);
        let mut l_fch: *mut f32 = (acts.fch)
            .offset(
                (l as libc::c_ulong)
                    .wrapping_mul(B)
                    .wrapping_mul(T)
                    .wrapping_mul(4 as libc::c_int as libc::c_ulong)
                    .wrapping_mul(C) as isize,
            );
        let mut l_fch_gelu: *mut f32 = (acts.fch_gelu)
            .offset(
                (l as libc::c_ulong)
                    .wrapping_mul(B)
                    .wrapping_mul(T)
                    .wrapping_mul(4 as libc::c_int as libc::c_ulong)
                    .wrapping_mul(C) as isize,
            );
        let mut l_fcproj: *mut f32 = (acts.fcproj)
            .offset(
                (l as libc::c_ulong).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C)
                    as isize,
            );
        let mut l_residual3: *mut f32 = (acts.residual3)
            .offset(
                (l as libc::c_ulong).wrapping_mul(B).wrapping_mul(T).wrapping_mul(C)
                    as isize,
            );
        layernorm_forward(
            l_ln1,
            l_ln1_mean,
            l_ln1_rstd,
            residual,
            l_ln1w,
            l_ln1b,
            B as libc::c_int,
            T as libc::c_int,
            C as libc::c_int,
        );
        matmul_forward(
            l_qkv,
            l_ln1,
            l_qkvw,
            l_qkvb,
            B as libc::c_int,
            T as libc::c_int,
            C as libc::c_int,
            (3 as libc::c_int as libc::c_ulong).wrapping_mul(C) as libc::c_int,
        );
        attention_forward(
            l_atty,
            l_preatt,
            l_att,
            l_qkv,
            B as libc::c_int,
            T as libc::c_int,
            C as libc::c_int,
            NH as libc::c_int,
        );
        matmul_forward(
            l_attproj,
            l_atty,
            l_attprojw,
            l_attprojb,
            B as libc::c_int,
            T as libc::c_int,
            C as libc::c_int,
            C as libc::c_int,
        );
        residual_forward(
            l_residual2,
            residual,
            l_attproj,
            B.wrapping_mul(T).wrapping_mul(C) as libc::c_int,
        );
        layernorm_forward(
            l_ln2,
            l_ln2_mean,
            l_ln2_rstd,
            l_residual2,
            l_ln2w,
            l_ln2b,
            B as libc::c_int,
            T as libc::c_int,
            C as libc::c_int,
        );
        matmul_forward(
            l_fch,
            l_ln2,
            l_fcw,
            l_fcb,
            B as libc::c_int,
            T as libc::c_int,
            C as libc::c_int,
            (4 as libc::c_int as libc::c_ulong).wrapping_mul(C) as libc::c_int,
        );
        gelu_forward(
            l_fch_gelu,
            l_fch,
            B
                .wrapping_mul(T)
                .wrapping_mul(4 as libc::c_int as libc::c_ulong)
                .wrapping_mul(C) as libc::c_int,
        );
        matmul_forward(
            l_fcproj,
            l_fch_gelu,
            l_fcprojw,
            l_fcprojb,
            B as libc::c_int,
            T as libc::c_int,
            (4 as libc::c_int as libc::c_ulong).wrapping_mul(C) as libc::c_int,
            C as libc::c_int,
        );
        residual_forward(
            l_residual3,
            l_residual2,
            l_fcproj,
            B.wrapping_mul(T).wrapping_mul(C) as libc::c_int,
        );
        l += 1;
        l;
    }
    residual = (acts.residual3)
        .offset(
            (((*model_consts).config.num_layers - 1 as libc::c_int) as libc::c_ulong)
                .wrapping_mul(B)
                .wrapping_mul(T)
                .wrapping_mul((*model_consts).config.channels as libc::c_ulong) as isize,
        );
    layernorm_forward(
        acts.lnf,
        acts.lnf_mean,
        acts.lnf_rstd,
        residual,
        params.lnfw,
        params.lnfb,
        B as libc::c_int,
        T as libc::c_int,
        (*model_consts).config.channels,
    );
    matmul_forward(
        acts.logits,
        acts.lnf,
        params.wte,
        0 as *const f32,
        B as libc::c_int,
        T as libc::c_int,
        (*model_consts).config.channels,
        Vp as libc::c_int,
    );
    softmax_forward(
        acts.probs,
        acts.logits,
        B as libc::c_int,
        T as libc::c_int,
        V as libc::c_int,
        Vp as libc::c_int,
    );
    if !targets.is_null() {
        crossentropy_forward(
            (*model).acts.losses,
            (*model).acts.probs,
            targets,
            B as libc::c_int,
            T as libc::c_int,
            Vp as libc::c_int,
        );
        let mut mean_loss: f32 = 0.0f32;
        let mut i_0: libc::c_int = 0 as libc::c_int;
        while (i_0 as libc::c_ulong) < B.wrapping_mul(T) {
            mean_loss += *((*model).acts.losses).offset(i_0 as isize);
            i_0 += 1;
            i_0;
        }
        (*model_consts).mean_loss = mean_loss / B.wrapping_mul(T) as f32;
    } else {
        (*model_consts).mean_loss = -1.0f32;
    }
    *res = (*model_consts).mean_loss;
}
#[no_mangle]
pub unsafe extern "C" fn gpt2_zero_grad(mut const_model: *mut GPT2Const) {
    if !((*const_model).grads_memory).is_null() {
        memset(
            (*const_model).grads_memory as *mut libc::c_void,
            0 as libc::c_int,
            ((*const_model).num_parameters)
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
    if !((*const_model).grads_acts_memory).is_null() {
        memset(
            (*const_model).grads_acts_memory as *mut libc::c_void,
            0 as libc::c_int,
            ((*const_model).num_activations)
                .wrapping_mul(::core::mem::size_of::<f32>() as libc::c_ulong),
        );
    }
}
#[no_mangle]
pub unsafe extern "C" fn gpt2_update(
    mut model: *mut GPT2,
    mut shadow_model: *mut GPT2,
    mut const_model: *mut GPT2Const,
    mut learning_rate: f32,
    mut beta1: f32,
    mut beta2: f32,
    mut eps: f32,
    mut weight_decay: f32,
    mut t: libc::c_int,
) {
    if ((*const_model).m_memory).is_null() {
        (*const_model)
            .m_memory = calloc(
            (*const_model).num_parameters,
            ::core::mem::size_of::<f32>() as libc::c_ulong,
        ) as *mut f32;
        (*const_model)
            .v_memory = calloc(
            (*const_model).num_parameters,
            ::core::mem::size_of::<f32>() as libc::c_ulong,
        ) as *mut f32;
    }
    let mut i: size_t = 0 as libc::c_int as size_t;
    while i < (*const_model).num_parameters {
        let mut param: f32 = *((*model).params_memory).offset(i as isize);
        let mut grad: f32 = *((*shadow_model).params_memory)
            .offset(i as isize);
        *((*shadow_model).params_memory)
            .offset(i as isize) = 0 as libc::c_int as f32;
        let mut m: f32 = beta1 * *((*const_model).m_memory).offset(i as isize)
            + (1.0f32 - beta1) * grad;
        let mut v: f32 = beta2 * *((*const_model).v_memory).offset(i as isize)
            + (1.0f32 - beta2) * grad * grad;
        let mut m_hat: f32 = m / (1.0f32 - powf(beta1, t as f32));
        let mut v_hat: f32 = v / (1.0f32 - powf(beta2, t as f32));
        if i == 0 as libc::c_int as libc::c_ulong {
            printf(
                b"grad: %f\n\0" as *const u8 as *const libc::c_char,
                grad as libc::c_double,
            );
            printf(
                b"old weight: %f\n\0" as *const u8 as *const libc::c_char,
                *((*model).params_memory).offset(i as isize) as libc::c_double,
            );
            fflush(stdout);
        }
        let mut update: f32 = learning_rate
            * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
        *((*const_model).m_memory).offset(i as isize) = m;
        *((*const_model).v_memory).offset(i as isize) = v;
        *((*model).params_memory).offset(i as isize) -= update;
        if i == 0 as libc::c_int as libc::c_ulong {
            printf(
                b"new weight: %f (update: %f)\n\0" as *const u8 as *const libc::c_char,
                *((*model).params_memory).offset(i as isize) as libc::c_double,
                update as libc::c_double,
            );
            fflush(stdout);
        }
        i = i.wrapping_add(1);
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn random_u32(mut state: *mut uint64_t) -> libc::c_uint {
    *state ^= *state >> 12 as libc::c_int;
    *state ^= *state << 25 as libc::c_int;
    *state ^= *state >> 27 as libc::c_int;
    return ((*state as libc::c_ulonglong)
        .wrapping_mul(0x2545f4914f6cdd1d as libc::c_ulonglong) >> 32 as libc::c_int)
        as libc::c_uint;
}
#[no_mangle]
pub unsafe extern "C" fn random_f32(mut state: *mut uint64_t) -> f32 {
    return (random_u32(state) >> 8 as libc::c_int) as f32 / 16777216.0f32;
}
#[no_mangle]
pub unsafe extern "C" fn sample_mult(
    mut probabilities: *mut f32,
    mut n: libc::c_int,
    mut coin: f32,
) -> libc::c_int {
    let mut cdf: f32 = 0.0f32;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < n {
        cdf += *probabilities.offset(i as isize);
        if coin < cdf {
            return i;
        }
        i += 1;
        i;
    }
    return n - 1 as libc::c_int;
}
unsafe fn main_0() -> libc::c_int {
    let mut model: GPT2 = GPT2 {
        params: ParameterTensors {
            wte: 0 as *mut f32,
            wpe: 0 as *mut f32,
            ln1w: 0 as *mut f32,
            ln1b: 0 as *mut f32,
            qkvw: 0 as *mut f32,
            qkvb: 0 as *mut f32,
            attprojw: 0 as *mut f32,
            attprojb: 0 as *mut f32,
            ln2w: 0 as *mut f32,
            ln2b: 0 as *mut f32,
            fcw: 0 as *mut f32,
            fcb: 0 as *mut f32,
            fcprojw: 0 as *mut f32,
            fcprojb: 0 as *mut f32,
            lnfw: 0 as *mut f32,
            lnfb: 0 as *mut f32,
        },
        param_sizes: [0; 16],
        params_memory: 0 as *mut f32,
        act_sizes: [0; 23],
        acts: ActivationTensors {
            encoded: 0 as *mut f32,
            ln1: 0 as *mut f32,
            ln1_mean: 0 as *mut f32,
            ln1_rstd: 0 as *mut f32,
            qkv: 0 as *mut f32,
            atty: 0 as *mut f32,
            preatt: 0 as *mut f32,
            att: 0 as *mut f32,
            attproj: 0 as *mut f32,
            residual2: 0 as *mut f32,
            ln2: 0 as *mut f32,
            ln2_mean: 0 as *mut f32,
            ln2_rstd: 0 as *mut f32,
            fch: 0 as *mut f32,
            fch_gelu: 0 as *mut f32,
            fcproj: 0 as *mut f32,
            residual3: 0 as *mut f32,
            lnf: 0 as *mut f32,
            lnf_mean: 0 as *mut f32,
            lnf_rstd: 0 as *mut f32,
            logits: 0 as *mut f32,
            probs: 0 as *mut f32,
            losses: 0 as *mut f32,
        },
        acts_memory: 0 as *mut f32,
    };
    let mut const_model: GPT2Const = GPT2Const {
        config: GPT2Config {
            max_seq_len: 0,
            vocab_size: 0,
            padded_vocab_size: 0,
            num_layers: 0,
            num_heads: 0,
            channels: 0,
        },
        grads: ParameterTensors {
            wte: 0 as *mut f32,
            wpe: 0 as *mut f32,
            ln1w: 0 as *mut f32,
            ln1b: 0 as *mut f32,
            qkvw: 0 as *mut f32,
            qkvb: 0 as *mut f32,
            attprojw: 0 as *mut f32,
            attprojb: 0 as *mut f32,
            ln2w: 0 as *mut f32,
            ln2b: 0 as *mut f32,
            fcw: 0 as *mut f32,
            fcb: 0 as *mut f32,
            fcprojw: 0 as *mut f32,
            fcprojb: 0 as *mut f32,
            lnfw: 0 as *mut f32,
            lnfb: 0 as *mut f32,
        },
        grads_memory: 0 as *mut f32,
        m_memory: 0 as *mut f32,
        v_memory: 0 as *mut f32,
        num_activations: 0,
        grads_acts: ActivationTensors {
            encoded: 0 as *mut f32,
            ln1: 0 as *mut f32,
            ln1_mean: 0 as *mut f32,
            ln1_rstd: 0 as *mut f32,
            qkv: 0 as *mut f32,
            atty: 0 as *mut f32,
            preatt: 0 as *mut f32,
            att: 0 as *mut f32,
            attproj: 0 as *mut f32,
            residual2: 0 as *mut f32,
            ln2: 0 as *mut f32,
            ln2_mean: 0 as *mut f32,
            ln2_rstd: 0 as *mut f32,
            fch: 0 as *mut f32,
            fch_gelu: 0 as *mut f32,
            fcproj: 0 as *mut f32,
            residual3: 0 as *mut f32,
            lnf: 0 as *mut f32,
            lnf_mean: 0 as *mut f32,
            lnf_rstd: 0 as *mut f32,
            logits: 0 as *mut f32,
            probs: 0 as *mut f32,
            losses: 0 as *mut f32,
        },
        grads_acts_memory: 0 as *mut f32,
        batch_size: 0,
        seq_len: 0,
        inputs: 0 as *mut libc::c_int,
        targets: 0 as *mut libc::c_int,
        mean_loss: 0.,
        num_parameters: 0,
    };
    let mut shadow_model: GPT2 = GPT2 {
        params: ParameterTensors {
            wte: 0 as *mut f32,
            wpe: 0 as *mut f32,
            ln1w: 0 as *mut f32,
            ln1b: 0 as *mut f32,
            qkvw: 0 as *mut f32,
            qkvb: 0 as *mut f32,
            attprojw: 0 as *mut f32,
            attprojb: 0 as *mut f32,
            ln2w: 0 as *mut f32,
            ln2b: 0 as *mut f32,
            fcw: 0 as *mut f32,
            fcb: 0 as *mut f32,
            fcprojw: 0 as *mut f32,
            fcprojb: 0 as *mut f32,
            lnfw: 0 as *mut f32,
            lnfb: 0 as *mut f32,
        },
        param_sizes: [0; 16],
        params_memory: 0 as *mut f32,
        act_sizes: [0; 23],
        acts: ActivationTensors {
            encoded: 0 as *mut f32,
            ln1: 0 as *mut f32,
            ln1_mean: 0 as *mut f32,
            ln1_rstd: 0 as *mut f32,
            qkv: 0 as *mut f32,
            atty: 0 as *mut f32,
            preatt: 0 as *mut f32,
            att: 0 as *mut f32,
            attproj: 0 as *mut f32,
            residual2: 0 as *mut f32,
            ln2: 0 as *mut f32,
            ln2_mean: 0 as *mut f32,
            ln2_rstd: 0 as *mut f32,
            fch: 0 as *mut f32,
            fch_gelu: 0 as *mut f32,
            fcproj: 0 as *mut f32,
            residual3: 0 as *mut f32,
            lnf: 0 as *mut f32,
            lnf_mean: 0 as *mut f32,
            lnf_rstd: 0 as *mut f32,
            logits: 0 as *mut f32,
            probs: 0 as *mut f32,
            losses: 0 as *mut f32,
        },
        acts_memory: 0 as *mut f32,
    };
    let mut const_shadow_model: GPT2Const = GPT2Const {
        config: GPT2Config {
            max_seq_len: 0,
            vocab_size: 0,
            padded_vocab_size: 0,
            num_layers: 0,
            num_heads: 0,
            channels: 0,
        },
        grads: ParameterTensors {
            wte: 0 as *mut f32,
            wpe: 0 as *mut f32,
            ln1w: 0 as *mut f32,
            ln1b: 0 as *mut f32,
            qkvw: 0 as *mut f32,
            qkvb: 0 as *mut f32,
            attprojw: 0 as *mut f32,
            attprojb: 0 as *mut f32,
            ln2w: 0 as *mut f32,
            ln2b: 0 as *mut f32,
            fcw: 0 as *mut f32,
            fcb: 0 as *mut f32,
            fcprojw: 0 as *mut f32,
            fcprojb: 0 as *mut f32,
            lnfw: 0 as *mut f32,
            lnfb: 0 as *mut f32,
        },
        grads_memory: 0 as *mut f32,
        m_memory: 0 as *mut f32,
        v_memory: 0 as *mut f32,
        num_activations: 0,
        grads_acts: ActivationTensors {
            encoded: 0 as *mut f32,
            ln1: 0 as *mut f32,
            ln1_mean: 0 as *mut f32,
            ln1_rstd: 0 as *mut f32,
            qkv: 0 as *mut f32,
            atty: 0 as *mut f32,
            preatt: 0 as *mut f32,
            att: 0 as *mut f32,
            attproj: 0 as *mut f32,
            residual2: 0 as *mut f32,
            ln2: 0 as *mut f32,
            ln2_mean: 0 as *mut f32,
            ln2_rstd: 0 as *mut f32,
            fch: 0 as *mut f32,
            fch_gelu: 0 as *mut f32,
            fcproj: 0 as *mut f32,
            residual3: 0 as *mut f32,
            lnf: 0 as *mut f32,
            lnf_mean: 0 as *mut f32,
            lnf_rstd: 0 as *mut f32,
            logits: 0 as *mut f32,
            probs: 0 as *mut f32,
            losses: 0 as *mut f32,
        },
        grads_acts_memory: 0 as *mut f32,
        batch_size: 0,
        seq_len: 0,
        inputs: 0 as *mut libc::c_int,
        targets: 0 as *mut libc::c_int,
        mean_loss: 0.,
        num_parameters: 0,
    };
    let mut tiny_stories_train: *const libc::c_char = b"dev/data/tinystories/TinyStories_train.bin\0"
        as *const u8 as *const libc::c_char;
    let mut tiny_stories_val: *const libc::c_char = b"dev/data/tinystories/TinyStories_val.bin\0"
        as *const u8 as *const libc::c_char;
    let mut tiny_shakespeare_train: *const libc::c_char = b"dev/data/tinyshakespeare/tiny_shakespeare_train.bin\0"
        as *const u8 as *const libc::c_char;
    let mut tiny_shakespeare_val: *const libc::c_char = b"dev/data/tinyshakespeare/tiny_shakespeare_val.bin\0"
        as *const u8 as *const libc::c_char;
    let mut train_tokens: *const libc::c_char = if access(
        tiny_shakespeare_train,
        0 as libc::c_int,
    ) != -(1 as libc::c_int)
    {
        tiny_shakespeare_train
    } else {
        tiny_stories_train
    };
    let mut val_tokens: *const libc::c_char = if access(
        tiny_shakespeare_val,
        0 as libc::c_int,
    ) != -(1 as libc::c_int)
    {
        tiny_shakespeare_val
    } else {
        tiny_stories_val
    };
    let mut B: size_t = 4 as libc::c_int as size_t;
    let mut T: size_t = 64 as libc::c_int as size_t;
    let mut train_loader: DataLoader = DataLoader {
        process_rank: 0,
        num_processes: 0,
        B: 0,
        T: 0,
        num_tokens: 0,
        shard_num_samples: 0,
        glob_result: glob_t {
            gl_pathc: 0,
            gl_pathv: 0 as *mut *mut libc::c_char,
            gl_offs: 0,
            gl_flags: 0,
            gl_closedir: None,
            gl_readdir: None,
            gl_opendir: None,
            gl_lstat: None,
            gl_stat: None,
        },
        current_shard_idx: 0,
        current_sample_idx: 0,
        tokens_file: 0 as *mut FILE,
        buffer: 0 as *mut uint16_t,
        inputs: 0 as *mut libc::c_int,
        targets: 0 as *mut libc::c_int,
        shuffle_rng: mt19937_state {
            seed_: 0,
            left_: 0,
            next_: 0,
            state_: [0; 624],
            MATRIX_A: [0; 2],
        },
        should_shuffle: 0,
        shard_indices: 0 as *mut libc::c_int,
        intra_shard_indices: 0 as *mut libc::c_int,
        total_batch_size_bytes: 0,
        local_batch_offset_bytes: 0,
        header_bytes: 0,
        file_size_bytes: 0,
    };
    let mut val_loader: DataLoader = DataLoader {
        process_rank: 0,
        num_processes: 0,
        B: 0,
        T: 0,
        num_tokens: 0,
        shard_num_samples: 0,
        glob_result: glob_t {
            gl_pathc: 0,
            gl_pathv: 0 as *mut *mut libc::c_char,
            gl_offs: 0,
            gl_flags: 0,
            gl_closedir: None,
            gl_readdir: None,
            gl_opendir: None,
            gl_lstat: None,
            gl_stat: None,
        },
        current_shard_idx: 0,
        current_sample_idx: 0,
        tokens_file: 0 as *mut FILE,
        buffer: 0 as *mut uint16_t,
        inputs: 0 as *mut libc::c_int,
        targets: 0 as *mut libc::c_int,
        shuffle_rng: mt19937_state {
            seed_: 0,
            left_: 0,
            next_: 0,
            state_: [0; 624],
            MATRIX_A: [0; 2],
        },
        should_shuffle: 0,
        shard_indices: 0 as *mut libc::c_int,
        intra_shard_indices: 0 as *mut libc::c_int,
        total_batch_size_bytes: 0,
        local_batch_offset_bytes: 0,
        header_bytes: 0,
        file_size_bytes: 0,
    };
    dataloader_init(
        &mut train_loader,
        train_tokens,
        B,
        T,
        0 as libc::c_int,
        1 as libc::c_int,
        1 as libc::c_int,
    );
    dataloader_init(
        &mut val_loader,
        val_tokens,
        B,
        T,
        0 as libc::c_int,
        1 as libc::c_int,
        0 as libc::c_int,
    );
    gpt2_build_from_checkpoint(
        &mut model,
        &mut const_model,
        b"gpt2_124M.bin\0" as *const u8 as *const libc::c_char,
    );
    gpt2_build_from_checkpoint(
        &mut shadow_model,
        &mut const_shadow_model,
        b"gpt2_124M.bin\0" as *const u8 as *const libc::c_char,
    );
    gpt2_init(&mut model, &mut const_model, B, T);
    gpt2_init(&mut shadow_model, &mut const_shadow_model, B, T);
    printf(
        b"train dataset num_batches: %zu\n\0" as *const u8 as *const libc::c_char,
        (train_loader.num_tokens).wrapping_div(B.wrapping_mul(T)),
    );
    printf(
        b"val dataset num_batches: %zu\n\0" as *const u8 as *const libc::c_char,
        (val_loader.num_tokens).wrapping_div(B.wrapping_mul(T)),
    );
    let mut val_num_batches: libc::c_int = 5 as libc::c_int;
    let mut tokenizer: Tokenizer = Tokenizer {
        vocab_size: 0,
        token_table: 0 as *mut *mut libc::c_char,
        init_ok: 0,
        eot_token: 0,
    };
    tokenizer_init(
        &mut tokenizer,
        b"gpt2_tokenizer.bin\0" as *const u8 as *const libc::c_char,
    );
    let mut rng_state: uint64_t = 1337 as libc::c_int as uint64_t;
    let mut gen_tokens: *mut libc::c_int = malloc_check(
        B
            .wrapping_mul(T)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
        b"/tmp/.tmpLh4A5T/source.c\0" as *const u8 as *const libc::c_char,
        1956 as libc::c_int,
    ) as *mut libc::c_int;
    let genT: libc::c_int = 64 as libc::c_int;
    let mut start: timespec = timespec { tv_sec: 0, tv_nsec: 0 };
    let mut end: timespec = timespec { tv_sec: 0, tv_nsec: 0 };
    let mut step: libc::c_int = 0 as libc::c_int;
    while step <= 40 as libc::c_int {
        if step % 10 as libc::c_int == 0 as libc::c_int {
            let mut val_loss: f32 = 0.0f32;
            dataloader_reset(&mut val_loader);
            let mut i: libc::c_int = 0 as libc::c_int;
            while i < val_num_batches {
                dataloader_next_batch(&mut val_loader);
                let mut res: f32 = 0.0f64 as f32;
                gpt2_forward(
                    &mut model,
                    &mut const_model,
                    val_loader.inputs,
                    val_loader.targets,
                    B,
                    T,
                    &mut res,
                );
                val_loss += const_model.mean_loss;
                i += 1;
                i;
            }
            val_loss /= val_num_batches as f32;
            printf(
                b"val loss %f\n\0" as *const u8 as *const libc::c_char,
                val_loss as libc::c_double,
            );
        }
        if step > 0 as libc::c_int && step % 20 as libc::c_int == 0 as libc::c_int {
            let mut i_0: libc::c_int = 0 as libc::c_int;
            while (i_0 as libc::c_ulong) < B.wrapping_mul(T) {
                *gen_tokens.offset(i_0 as isize) = tokenizer.eot_token;
                i_0 += 1;
                i_0;
            }
            printf(b"generating:\n---\n\0" as *const u8 as *const libc::c_char);
            let mut t: libc::c_int = 1 as libc::c_int;
            while t < genT {
                let mut res_0: f32 = 0.0f64 as f32;
                gpt2_forward(
                    &mut model,
                    &mut const_model,
                    gen_tokens,
                    0 as *mut libc::c_int,
                    B,
                    T,
                    &mut res_0,
                );
                let mut probs: *mut f32 = (model.acts.probs)
                    .offset(
                        ((t - 1 as libc::c_int) * const_model.config.padded_vocab_size)
                            as isize,
                    );
                let mut coin: f32 = random_f32(&mut rng_state);
                let mut next_token: libc::c_int = sample_mult(
                    probs,
                    const_model.config.vocab_size,
                    coin,
                );
                *gen_tokens.offset(t as isize) = next_token;
                if tokenizer.init_ok != 0 {
                    let mut token_str: *const libc::c_char = tokenizer_decode(
                        &mut tokenizer,
                        next_token as uint32_t,
                    );
                    safe_printf(token_str);
                } else {
                    printf(b"%d \0" as *const u8 as *const libc::c_char, next_token);
                }
                fflush(stdout);
                t += 1;
                t;
            }
            printf(b"\n---\n\0" as *const u8 as *const libc::c_char);
        }
        clock_gettime(1 as libc::c_int, &mut start);
        dataloader_next_batch(&mut train_loader);
        printf(b"before __enzyme_autodiff\n\0" as *const u8 as *const libc::c_char);
        fflush(stdout);
        let mut res_1: f32 = 0.0f64 as f32;
        let mut dres: f32 = 1.0f64 as f32;
        
        //__enzyme_autodiff(
        //    ::core::mem::transmute::<
        //        Option::<
        //            unsafe extern "C" fn(
        //                *mut GPT2,
        //                *mut GPT2Const,
        //                *mut libc::c_int,
        //                *mut libc::c_int,
        //                size_t,
        //                size_t,
        //                *mut f32,
        //            ) -> (),
        //        >,
        //        *mut libc::c_void,
        //    >(
        //        Some(
        //            gpt2_forward
        //                as unsafe extern "C" fn(
        //                    *mut GPT2,
        //                    *mut GPT2Const,
        //                    *mut libc::c_int,
        //                    *mut libc::c_int,
        //                    size_t,
        //                    size_t,
        //                    *mut f32,
        //                ) -> (),
        //        ),
        //    ),
        //    enzyme_dup,
        //    &mut model as *mut GPT2,
        //    &mut shadow_model as *mut GPT2,
        //    enzyme_const,
        //    &mut const_model as *mut GPT2Const,
        //    enzyme_const,
        //    train_loader.inputs,
        //    enzyme_const,
        //    train_loader.targets,
        //    enzyme_const,
        //    B,
        //    enzyme_const,
        //    T,
        //    enzyme_dupnoneed,
        //    &mut res_1 as *mut f32,
        //    &mut dres as *mut f32,
        //);
        gpt2_ad_backward(&mut model as *mut GPT2, &mut shadow_model as *mut GPT2, &mut const_model as *mut GPT2Const, train_loader.inputs, train_loader.targets, B, T, &mut res_1, &mut dres);
        println!("res: {res_1}");
        println!("dres: {dres}");
        println!("grad: {}",unsafe {*shadow_model.params.wte.wrapping_add(0)});
        printf(b"after __enzyme_autodiff\n\0" as *const u8 as *const libc::c_char);
        fflush(stdout);
        gpt2_update(
            &mut model,
            &mut shadow_model,
            &mut const_model,
            1e-4f32,
            0.9f32,
            0.999f32,
            1e-8f32,
            0.0f32,
            step + 1 as libc::c_int,
        );
        printf(b"checkpoint\0" as *const u8 as *const libc::c_char);
        fflush(stdout);
        clock_gettime(1 as libc::c_int, &mut end);
        let mut time_elapsed_s: libc::c_double = (end.tv_sec - start.tv_sec)
            as libc::c_double + (end.tv_nsec - start.tv_nsec) as libc::c_double / 1e9f64;
        printf(
            b"step %d: train loss %f (took %f ms)\n\0" as *const u8
                as *const libc::c_char,
            step,
            const_model.mean_loss as libc::c_double,
            time_elapsed_s * 1000 as libc::c_int as libc::c_double,
        );
        step += 1;
        step;
    }
    dataloader_free(&mut train_loader);
    dataloader_free(&mut val_loader);
    tokenizer_free(&mut tokenizer);
    free(gen_tokens as *mut libc::c_void);
    return 0 as libc::c_int;
}
pub fn main() {
    unsafe { ::std::process::exit(main_0() as i32) }
}

