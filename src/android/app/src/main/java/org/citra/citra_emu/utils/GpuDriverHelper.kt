// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

package org.citra.citra_emu.utils

import android.net.Uri
import android.os.Build
import androidx.documentfile.provider.DocumentFile
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.lang.IllegalStateException
import java.util.zip.ZipEntry
import java.util.zip.ZipException
import java.util.zip.ZipInputStream
import org.citra.citra_emu.BuildConfig
import org.citra.citra_emu.CitraApplication
import org.citra.citra_emu.NativeLibrary
import org.citra.citra_emu.utils.FileUtil.asDocumentFile
import org.citra.citra_emu.utils.FileUtil.inputStream

object GpuDriverHelper {
    private const val META_JSON_FILENAME = "meta.json"

    // Where a build that comes with a driver describes it; the driver itself is one of the app's
    // native libraries.
    private const val BUILT_IN_DRIVER_META_ASSET = "gpu_driver/meta.json"

    // In the driver installation directory: the system's driver was chosen. Without it, and
    // without a driver installed there, the one the app comes with is used.
    private const val SYSTEM_DRIVER_FILENAME = "use_system_driver"

    // In the app's files: how the trial of the driver the app comes with went.
    private const val BUILT_IN_DRIVER_TRIAL_FILENAME = "built_in_driver"

    // In place of a driver package's location.
    val SYSTEM_DRIVER: Uri = Uri.EMPTY
    val BUILT_IN_DRIVER: Uri = Uri.parse("driver:built-in")

    private var isBuiltInDriverUsed = false
    private var fileRedirectionPath: String? = null
    var driverInstallationPath: String? = null
    private var hookLibPath: String? = null

    val driverStoragePath: DocumentFile
        get() {
            // Bypass directory initialization checks
            val root = DocumentFile.fromTreeUri(
                CitraApplication.appContext,
                Uri.parse(DirectoryInitialization.userPath)
            )!!
            var driverDirectory = root.findFile("gpu_drivers")
            if (driverDirectory == null) {
                driverDirectory = FileUtil.createDir(root.uri.toString(), "gpu_drivers")
            }
            return driverDirectory!!
        }

    fun initializeDriverParameters() {
        try {
            // Initialize the file redirection directory.
            fileRedirectionPath =
                DirectoryInitialization.internalUserPath + "/gpu/vk_file_redirect/"

            // Initialize the driver installation directory.
            driverInstallationPath = CitraApplication.appContext
                .filesDir.canonicalPath + "/gpu_driver/"
        } catch (e: IOException) {
            throw RuntimeException(e)
        }

        // Initialize directories.
        initializeDirectories()

        // Profiling builds take a driver package left for them, so that driver builds can be
        // tried without going through the driver manager.
        if (BuildConfig.BUILD_TYPE == "profile") {
            installTestDriver()
        }

        // Initialize hook libraries directory.
        hookLibPath = CitraApplication.appContext.applicationInfo.nativeLibraryDir + "/"

        // Initialize GPU driver: the one that was installed, the one the app comes with, or the
        // system's.
        isBuiltInDriverUsed = initializeBuiltInDriver()
        if (!isBuiltInDriverUsed) {
            NativeLibrary.initializeGpuDriver(
                hookLibPath,
                driverInstallationPath,
                customDriverData.libraryName,
                fileRedirectionPath,
                false
            )
        }
    }

    /**
     * Uses the driver the app comes with, unless another was chosen or it does not work here. It
     * has to show that it works once per build of the app and of the system, and the outcome is
     * kept: a trial that is still noted as running when the app starts is one the app did not
     * survive.
     */
    private fun initializeBuiltInDriver(): Boolean {
        val driver = builtInDriverData ?: return false
        if (customDriverData.name != null ||
            File(driverInstallationPath + SYSTEM_DRIVER_FILENAME).exists()
        ) {
            return false
        }

        val build = "${BuildConfig.VERSION_CODE} ${Build.FINGERPRINT}"
        val trial = File(CitraApplication.appContext.filesDir, BUILT_IN_DRIVER_TRIAL_FILENAME)
        val outcome = try {
            trial.readText()
        } catch (_: IOException) {
            ""
        }
        if (outcome == "fails $build" || outcome == "running $build") {
            Log.warning("[GpuDriverHelper] ${driver.name} is not used, earlier trial: $outcome")
            return false
        }

        val isProven = outcome == "works $build"
        if (!isProven) {
            // On the storage before the driver is touched, in case the device does not survive it
            FileOutputStream(trial).use {
                it.write("running $build".toByteArray())
                it.fd.sync()
            }
        }
        val works = NativeLibrary.initializeGpuDriver(
            hookLibPath,
            hookLibPath,
            driver.libraryName,
            fileRedirectionPath,
            !isProven
        )
        if (!isProven) {
            trial.writeText((if (works) "works " else "fails ") + build)
        }
        Log.info("[GpuDriverHelper] The app's own driver ${driver.name}: taken $works")
        return works
    }

    /**
     * Installs test_driver.zip from the app's external files folder (where adb can put it:
     * /sdcard/Android/data/<package>/files) in place of the current driver, once: the file is
     * removed afterwards. An empty file of that name goes back to the system driver, one that is
     * no package to the driver the app comes with, which gets another trial.
     */
    private fun installTestDriver() {
        val zip = File(CitraApplication.appContext.getExternalFilesDir(null), "test_driver.zip")
        if (!zip.isFile) {
            return
        }
        try {
            val installation = File(driverInstallationPath!!)
            installation.deleteRecursively()
            installation.mkdirs()
            File(CitraApplication.appContext.filesDir, BUILT_IN_DRIVER_TRIAL_FILENAME).delete()
            if (zip.length() > 0) {
                FileUtil.unzipToInternalStorage(
                    BufferedInputStream(zip.inputStream()),
                    installation
                )
            } else {
                File(installation, SYSTEM_DRIVER_FILENAME).createNewFile()
            }
            Log.info("[GpuDriverHelper] Test driver of ${zip.length()} bytes installed")
        } catch (e: Exception) {
            Log.error("[GpuDriverHelper] Test driver ${zip.path} not installed: $e")
        }
        zip.delete()
    }

    fun getDrivers(): MutableList<Pair<Uri, GpuDriverMetadata>> {
        val driverZips = driverStoragePath.listFiles()
        val drivers: MutableList<Pair<Uri, GpuDriverMetadata>> =
            driverZips
                .filter { it.isFile }
                .mapNotNull {
                    val metadata = getMetadataFromZip(it.inputStream())
                    metadata.name?.let { _ -> Pair(it.uri, metadata) }
                }
                .sortedByDescending { it: Pair<Uri, GpuDriverMetadata> -> it.second.name }
                .distinct()
                .toMutableList()

        // TODO: Get system driver information
        drivers.add(0, Pair(SYSTEM_DRIVER, GpuDriverMetadata()))
        // The default comes first
        builtInDriverData?.let { drivers.add(0, Pair(BUILT_IN_DRIVER, it)) }
        return drivers
    }

    /**
     * Goes back to the default: the driver the app comes with, if it works here, or the system's.
     */
    fun installDefaultDriver() {
        File(driverInstallationPath!!).deleteRecursively()
        initializeDriverParameters()
    }

    /**
     * The default, when it was asked for: a driver that did not work gets another trial.
     */
    fun installBuiltInDriver() {
        File(CitraApplication.appContext.filesDir, BUILT_IN_DRIVER_TRIAL_FILENAME).delete()
        installDefaultDriver()
    }

    fun installSystemDriver() {
        val installation = File(driverInstallationPath!!)
        installation.deleteRecursively()
        installation.mkdirs()
        File(installation, SYSTEM_DRIVER_FILENAME).createNewFile()
        initializeDriverParameters()
    }

    fun copyDriverToExternalStorage(driverUri: Uri): DocumentFile? {
        // Ensure we have directories.
        initializeDirectories()

        // Copy the zip file URI to user data
        val copiedFile =
            FileUtil.copyToExternalStorage(driverUri, driverStoragePath) ?: return null

        // Validate driver
        val metadata = getMetadataFromZip(copiedFile.inputStream())
        if (metadata.name == null) {
            copiedFile.delete()
            return null
        }

        if (metadata.minApi > Build.VERSION.SDK_INT) {
            copiedFile.delete()
            return null
        }
        return copiedFile
    }

    /**
     * Copies driver zip into user data directory so that it can be exported along with
     * other user data and also unzipped into the installation directory
     */
    fun installCustomDriverComplete(driverUri: Uri): Boolean {
        // Revert to the default in the event the specified driver is bad.
        installDefaultDriver()

        // Ensure we have directories.
        initializeDirectories()

        // Copy the zip file URI to user data
        val copiedFile =
            FileUtil.copyToExternalStorage(driverUri, driverStoragePath) ?: return false

        // Validate driver
        val metadata = getMetadataFromZip(copiedFile.inputStream())
        if (metadata.name == null) {
            copiedFile.delete()
            return false
        }

        if (metadata.minApi > Build.VERSION.SDK_INT) {
            copiedFile.delete()
            return false
        }

        // Unzip the driver.
        try {
            FileUtil.unzipToInternalStorage(
                BufferedInputStream(copiedFile.inputStream()),
                File(driverInstallationPath!!)
            )
        } catch (e: SecurityException) {
            return false
        }

        // Initialize the driver parameters.
        initializeDriverParameters()

        return true
    }

    /**
     * Unzips driver into private installation directory
     */
    fun installCustomDriverPartial(driver: Uri): Boolean {
        // Revert to the default in the event the specified driver is bad.
        installDefaultDriver()

        // Ensure we have directories.
        initializeDirectories()

        // Validate driver
        val metadata = getMetadataFromZip(driver.inputStream())
        if (metadata.name == null) {
            driver.asDocumentFile()?.delete()
            return false
        }

        // Unzip the driver to the private installation directory
        try {
            FileUtil.unzipToInternalStorage(
                BufferedInputStream(driver.inputStream()),
                File(driverInstallationPath!!)
            )
        } catch (e: SecurityException) {
            return false
        }

        // Initialize the driver parameters.
        initializeDriverParameters()

        return true
    }

    /**
     * Takes in a zip file and reads the meta.json file for presentation to the UI
     *
     * @param driver Zip containing driver and meta.json file
     * @return A non-null [GpuDriverMetadata] instance that may have null members
     */
    fun getMetadataFromZip(driver: InputStream): GpuDriverMetadata {
        try {
            ZipInputStream(driver).use { zis ->
                var entry: ZipEntry? = zis.nextEntry
                while (entry != null) {
                    if (!entry.isDirectory && entry.name.lowercase().contains(".json")) {
                        val size = if (entry.size == -1L) 0L else entry.size
                        return GpuDriverMetadata(zis, size)
                    }
                    entry = zis.nextEntry
                }
            }
        } catch (_: ZipException) {
        }
        return GpuDriverMetadata()
    }

    external fun supportsCustomDriverLoading(): Boolean

    // True when the device has Qualcomm's KGSL GPU driver.
    external fun isAdrenoGpu(): Boolean

    // Parse the custom driver metadata to retrieve the name.
    val customDriverData: GpuDriverMetadata
        get() = GpuDriverMetadata(File(driverInstallationPath + META_JSON_FILENAME))

    // The driver the app was built with, for whatever device; none if it was built without.
    val packagedDriverData: GpuDriverMetadata? by lazy {
        try {
            CitraApplication.appContext.assets.open(BUILT_IN_DRIVER_META_ASSET)
                .use { GpuDriverMetadata(it, 0) }
                .takeIf { it.name != null && it.libraryName != null }
        } catch (_: IOException) {
            null
        }
    }

    // That driver, if it is for this device.
    val builtInDriverData: GpuDriverMetadata? by lazy {
        packagedDriverData?.takeIf {
            it.minApi <= Build.VERSION.SDK_INT &&
                supportsCustomDriverLoading() &&
                isAdrenoGpu() &&
                File(
                    CitraApplication.appContext.applicationInfo.nativeLibraryDir,
                    it.libraryName!!
                ).isFile
        }
    }

    // The driver that is used; without a name it is the system's.
    val currentDriverData: GpuDriverMetadata
        get() = if (isBuiltInDriverUsed) builtInDriverData!! else customDriverData

    /**
     * Whether this entry of the list of drivers is the one that is used. The app can come with a
     * driver whose package was also installed: the two have the same metadata.
     */
    fun isCurrentDriver(driver: Pair<Uri, GpuDriverMetadata>): Boolean =
        isBuiltInDriverUsed == (driver.first == BUILT_IN_DRIVER) &&
            driver.second == currentDriverData

    fun initializeDirectories() {
        // Ensure the file redirection directory exists.
        val fileRedirectionDir = File(fileRedirectionPath!!)
        if (!fileRedirectionDir.exists()) {
            fileRedirectionDir.mkdirs()
        }
        // Ensure the driver installation directory exists.
        val driverInstallationDir = File(driverInstallationPath!!)
        if (!driverInstallationDir.exists()) {
            driverInstallationDir.mkdirs()
        }
        // Ensure the driver storage directory exists
        if (!driverStoragePath.exists()) {
            throw IllegalStateException("Driver storage directory couldn't be created!")
        }
    }
}
