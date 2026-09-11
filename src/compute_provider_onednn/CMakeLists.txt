# The factor remains header-only until its AUDITED product implementation is
# added. Keeping the target real lets the component/product graph validate the
# new semantic edge without probing or downloading oneDNN in this transition.
hyle_add_module(compute_provider_onednn)
