# AQTp : Accurate Quantized Training for production

go/aqtp

## AQT Users chat

go/aqt-chat

This is a good place to ask any questions about quantization in JAX and AQT usage.
(Consider starting threads with **bolded title**).

Note: If for some reason you see a message 'Failed to join', that happened before for unknown reasons. Please ping yichizh@ or lew@ and we will add you manually.



## Main README.md

Most of the AQT documentation is in the [main README.md](../README.md). It
contains:

-   installation instructions (`pip install aqtp`) ,
-   list of features,
-   links to Cloud ML blog posts and research papers,
-   tutorial on how to use it with JAX and Flax libraries,
-   the AQT configuration,
-   relation of different AQT versions (v1, v2, TF, ...)
-   simplified description of the internal AQT math.

## Blog post

[Cloud ML AQT introduction](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e/)
is a well-polished general introduction to AQT.

## AQT goals

AQT is a JAX library has two primary goals:

1.  Acceleration of JAX computation and memory transfers using int8, int4,
    float8 and other numerical formats available on the existing hardware.
1.  Research on quantization to aid design of future Google TPUs.

## AQT value

AQT-int8 allows typically a 20-40% speedup in training time to a solution of
equal quality while outputting a pre-quantized model on platforms with 8-bit
acceleration (VLC and newer). AQT achieves this by evaluating most of the
matmuls in int8 (both fwd pass and bwd pass).

AQT is easy to use with a drop in lax.dot_general replacement and existing
integrations into Flax, Pax and MaxText. AQT was announced at Google Cloud Next,
and has existing adoption with Cloud customers. AQT delivered a key 39.9%
speedup in MLPerf training 2023.

AQT training acceleration is applicable to both pre-training and fine-tuning
yielding similarly good results.

[Cloud ML AQT introduction](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e/)
has some more quality details.

## Google3 support

AQT is a rather quickly evolving project with a wide variety of users.
When we do a refactoring that affects other project, we will do our best to
update all of google3 code. If you'd like your project to be officially supported please consider:

- Provide some basic tests that we can add to AQT presubmit. Here is [an example](http://google3/third_party/py/aqt/METADATA;l=51;rcl=596716629).
- Add `mdb-group:aqt-buildcops` line to `OWNERS` of your project. This will be used *only* for the AQT maintenance CLs.

If that recipe is infeasible for some reason, please contact `lew@`.

When submitting code to AQT that is not a pure refactoring, flaky test is encouraged
to run in order to update the training loss in flax e2e example. The command is

```
blaze  test --test_filter=MnistTest --test_output=errors //third_party/py/aqt/jax/v2/examples:flax_e2e_model_test --runs_per_test_detects_flakes --runs_per_test=50
```


## How to use AQT in Pax

We don't have a well developed documentation in Pax, but at least we can leave
the reader with some example links:

-   This
    [`cfg_set_quantization(...)`](http://google3/nlp/mum/pax/quantization/experiments.py;l=50;rcl=584723066)
    demonstrates how to inject AQT quantization into one particular model.
-   This function
    [`cfg_set_quantization(...)`](http://google3/intelligence/mobile_llms/pax/ulm/pretrain.py;l=354;rcl=584722680)
    demonstrates how to use go/fiddle selectors to make it a bit more automatix.

We will try extend a documentation whenever questions appear on AQT Users chat.


## How to use AQT in Gemax

[Gemax quantization documentation](https://g3doc.corp.google.com/learning/gemini/gemax/g3doc/quantization.md?cl=head)

## Research with AQT

We are constantly working to make AQTv2 a modular library.
The aim is to make it easy to add new quantization and sparsity algorithms.

For instance one can implement custom calibrations instead of
[AbsMax](https://source.corp.google.com/piper///depot/google3/third_party/py/aqt/jax/v2/calibration.py).
(We are just waiting for an opportunity to port
[AQTv1 excellent calibrations](http://google3/third_party/py/aqt/jax/aqt_tensor.py;l=169;rcl=568584281)
to this API.)

One can implement custom weight/activation representation as well. This is
[general int numerics](https://source.corp.google.com/piper///depot/google3/third_party/py/aqt/jax/v2/numerics/int_numerics.py).
*float8* will be submitted soon.

There are many algorithms that do not fit into these two simple abstractions,
but we will extend AQT to be universal quantization glue.

## AQT Infrastructure and AQT Algorithms

Making AQT more modular leads to a situation where AQT factors
itself into two part.
[This presentation](https://docs.google.com/presentation/d/1vxO_EUNfCO9oGkFQZGqRxf97BSWInJ2neVFSzN4RqEY/edit#slide=id.p)
has much more details.

### AQT Infrastructure

Turns out that the most of AQT code is infrastructure:

-   attaches custom gradient,
-   handles passing configuration,
-   handles passing state from Flax/Pax/Gemax to pure JAX algos,
-   implements complex transpositions needed for dot_general gradient,
-   and more.

It might make sense to integrate this part into JAX/Flax (some code was copied from there).

### AQT Algorithms

The successful AQT algorithms are simple equations. They are encoded in
calibration and numerics linked in [section above](#research-with-aqt).
This is fairly small amount of code, making AQT easy to use for researchers.
The added benefit is being able to easily try new algorithms on
production models that adopted AQT.

## AQT Development

Please refer to [this page](./aqt_development.md)

## AQT Catalyst team

AQT is developed by
[AQT Catalyst team](https://moma.corp.google.com/team/1448391999960).

All our presentations are here: go/catalyst-presentations.
