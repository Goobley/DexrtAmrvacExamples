## Example AMRVAC -> Sparse DexRT Atmosphere Processing

Here is a couple of examples of the current state of the art in model prep,
using the recently developed sparse input models for DexRT. These are not
checked closely when loaded (essentially dumped straight into the target
arrays), so require careful manipulation in the Python. Both of the scripts here
can output dense and sparse atmospheres (that can be rehydrated by the `dexrt`
tools), for comparison of the process. An additional complication to be mindful
of is array ordering (amrvac `[x, y, z]`, dexrt `[z, y, x]`, and the meaning of
these axes: y-up for amrvac and z-up for dexrt.)