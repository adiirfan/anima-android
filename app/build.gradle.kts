import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.anima.npu"
    compileSdk = 37
    ndkVersion = "28.2.13676358"
    defaultConfig {
        applicationId = "com.anima.npu"
        minSdk = 31
        targetSdk = 36
        // Build time in seconds: grows with every build and doubles as Android's version code (limit 2.1e9,
        // reached in 2036). publish-app.sh passes it as -PappBuild; a plain build gets 0 and never offers itself.
        val appBuild = (project.findProperty("appBuild") as String?)?.toLong() ?: 0L
        versionCode = if (appBuild > 0) appBuild.toInt() else 1
        buildConfigField("long", "APP_BUILD", "${appBuild}L")
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
        // Qualcomm's QAIRT folder (for the QNN headers): qnn.sdk.dir in local.properties, or QNN_SDK_ROOT
        val qnnSdk = Properties().apply {
            rootProject.file("local.properties").takeIf { it.isFile }?.inputStream()?.use { load(it) }
        }.getProperty("qnn.sdk.dir") ?: System.getenv("QNN_SDK_ROOT")
        externalNativeBuild { cmake {
            arguments += "-DANDROID_STL=c++_shared"
            if (qnnSdk != null) arguments += "-DQNN_SDK=$qnnSdk"
        } }
    }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt") } }
    // The neural unit only loads Qualcomm's libraries from a real folder on disk; packed inside the
    // app they are silently refused and nothing runs on it.
    packaging { jniLibs { useLegacyPackaging = true } }
    buildTypes { release { isMinifyEnabled = false } }
    buildFeatures { compose = true; buildConfig = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom-alpha:2026.07.01"))
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.11.0")
    implementation("androidx.core:core-ktx:1.13.1")   // FileProvider for app updates
    implementation("ai.djl.huggingface:tokenizers:0.33.0")
    implementation("ai.djl.android:tokenizer-native:0.33.0")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")   // Android's org.json is a stub in JVM tests
}
