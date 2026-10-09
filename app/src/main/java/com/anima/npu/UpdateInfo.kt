package com.anima.npu

import org.json.JSONObject

/** latest.json in the releases repo: the newest published build of the app. */
data class UpdateInfo(val build: Long, val size: Long, val sha256: String, val notes: String) {
    /** An unstamped build (0, made without publish-app.sh) is a development build: it never offers updates. */
    fun isNewerThan(current: Long): Boolean = current > 0 && build > current

    companion object {
        fun parse(json: String): UpdateInfo {
            val o = JSONObject(json)
            return UpdateInfo(o.getLong("build"), o.getLong("size"), o.getString("sha256"), o.optString("notes", ""))
        }
    }
}
