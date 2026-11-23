# Azahar RF

**Azahar RF is an experimental fork of Azahar, the Nintendo 3DS emulator. It targets Adreno GPUs only.**

- **Adreno only:** it is made for Android devices with a Qualcomm Adreno GPU. Other GPUs are neither supported nor tested.
- **Experimental:** expect bugs, and things that change from one build to the next.
- **Not an official Azahar release:** please don't report problems with it to the Azahar project.

Builds are on the [Releases](https://github.com/rfandango/azahar/releases) page. They come with a Vulkan driver for Adreno GPUs, based on Mesa's Turnip, and install next to Azahar.

*Azahar RF is not affiliated with or endorsed by Nintendo or the Azahar project.*

---

The rest of this file is Azahar's own README, cut down to Android. Its downloads, requirements and links are for Azahar, not for this fork.

---

![Azahar Emulator](https://azahar-emu.org/resources/images/logo/azahar-name-and-logo.svg)

![Google Play Downloads](https://playbadges.pavi2410.com/badge/downloads?id=io.github.lime3ds.android&pretty&label=Play%20Store%20Downloads)

**Azahar** is a free and open-source high level Nintendo 3DS emulator. Our goal is to give 3DS owners a place to enjoy their library of titles with improvements to the original hardware, such as higher resolutions, modern controllers or save states. The emulator also serves as a debugging hub for homebrew developers and as a research and preservation platform for the 3DS ecosystem.

The project continues the legacy of **Citra** and is actively developed by a community of contributors.

*Azahar is not affiliated with or endorsed by Nintendo.*

# Installation

There are two variants of Azahar available on Android, those being the Vanilla and Google Play builds.

The Vanilla build is technically superior, as it uses an alternative method of file management which is faster, but isn't permitted on the Google Play store.

For most users, we currently recommended downloading Azahar on Android via the Google Play Store for ease of accessibility:

<a href='https://play.google.com/store/apps/details?id=io.github.lime3ds.android'><img width='180' alt='Get it on Google Play' src='https://raw.githubusercontent.com/pioug/google-play-badges/06ccd9252af1501613da2ca28eaffe31307a4e6d/svg/English.svg'/></a>

# How can I contribute?

### Pull requests

If you want to implement a change and have the technical capability to do so, we would be happy to accept your contributions.

If you are contributing a new feature, it is highly suggested that you first make a Feature Request issue to discuss the addition before writing any code. This is to ensure that your time isn't wasted working on a feature which isn't deemed appropriate for the project.

After creating a pull request, please don't repeatedly merge `master` into your branch. A maintainer will update the branch for you if/ when it is appropriate to do so.

### Language translations

Additionally, we are accepting language translations on [Transifex](https://app.transifex.com/azahar/azahar). If you know a non-english language listed on our Transifex page, please feel free to contribute.

> [!NOTE]
> We are not currently accepting new languages for translation. Please do not request for new languages or language variants to be added.

# Minimum requirements

Below are the minimum requirements to run Azahar:

```
Operating System: Android 10.0+ (64-bit)
CPU: Snapdragon 835 SoC or better
GPU: OpenGL ES 3.2 or Vulkan 1.1 support
Memory: 2GB of RAM. 4GB is recommended
```
