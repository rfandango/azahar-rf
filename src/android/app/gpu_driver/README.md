# A GPU driver in the app

The app can come with a Vulkan driver, which it then uses by default on Adreno GPUs. The driver is
put in when the app is built and is not part of the source: git ignores everything in this folder
but this file.

It is taken from a driver package, the same zip the app's driver manager installs: a `meta.json`
and the library that its `libraryName` names. Put one package in this folder, or name it:

    ./gradlew assembleVanillaRelease -PgpuDriver=/path/to/driver.zip

- `-PgpuDriver` goes before this folder. It can also stand as `gpuDriver=...` in
  `~/.gradle/gradle.properties`, or in the environment as `ORG_GRADLE_PROJECT_gpuDriver`.
- `-PgpuDriver=none` builds without a driver. So does an empty folder.
- More than one package here is an error: leave one, or name it.

The build says which driver went in (`GPU driver in the app: ...`). The driver has to be Mesa's
Turnip for arm64; another one is refused on the phone, and the system's is used.

Which driver the app uses (`GpuDriverHelper.kt`), in this order:

1. The one chosen in the driver manager: a package, or the system's driver.
2. This one, if the phone has an Adreno GPU, the Android version is the one `minApi` asks for and
   the driver works. It is tried once per build of the app and of the system, by making a device
   with it (`driver_probe.cpp`), and the outcome is kept. If the app does not survive that trial,
   the system's driver is used from then on. Choosing this driver in the driver manager tries it
   again.
3. The system's driver.

A driver with another build id makes the games' stored pipelines be built again once. Mesa's
licence is on the app's licence screen in builds that come with a driver.
