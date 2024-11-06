project('tsinfer', 'c')
 
tskit_proj = subproject('tskit')
tskit_dep = tskit_proj.get_variable('tskit_dep')

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required : false)
cunit_dep = dependency('cunit')

extra_c_args = [
    '-std=c11', '-Wall', '-Wextra', '-Werror', '-Wpedantic', '-W',
    '-Wmissing-prototypes',  '-Wstrict-prototypes',
    '-Wconversion', '-Wshadow', '-Wpointer-arith', '-Wcast-align',
    '-Wcast-qual', '-Wwrite-strings', '-Wnested-externs',
    '-fshort-enums', '-fno-common']

tsinfer_sources =[
    'ancestor_matcher.c', 'ancestor_builder.c', 'tree_sequence_builder.c',
    'object_heap.c', 'err.c']

avl_lib = static_library('avl', sources: ['avl.c'])
tsinfer_lib = static_library('tsinfer', 
    sources: tsinfer_sources, dependencies: [m_dep, tskit_dep], 
    c_args: extra_c_args, link_with:[avl_lib])

unit_tests = executable('tests', 
    sources: ['tests/tests.c'], 
    link_with: [tsinfer_lib], dependencies:[cunit_dep, tskit_dep])
test('Unit tests', unit_tests)
