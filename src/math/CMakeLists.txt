# math: scalar helpers (conventions: modernization of clay math/utils.h). Header-
# only -- div_ceil/round_up_to_multiple/sgn/approx_eq are all inline constexpr
# templates. INTERFACE library: carries the include dir, no .so/.lib of its own.
# (clay's bit tricks/clamp/pow/consts are superseded by <bit>/<algorithm>/<cmath>/
# <numbers>; only the genuine increments live in math/scalar.h.)
hyle_add_module(math)
