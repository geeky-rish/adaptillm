package com.example.adaptillm.core

import android.app.ActivityManager
import android.content.Context
import android.os.BatteryManager

/**
 * DeviceMonitor — Real-time system resource tracking.
 *
 * Battery measurement strategy:
 *   - getBatteryLevel(): coarse % (0-100), for UI display
 *   - getBatteryMicroAmps(): instantaneous current draw in µA
 *   - computeEnergyProxy(): µA × ms = µA·ms, proportional to energy
 *
 * Why µA instead of %:
 *   Battery % has 1% granularity → delta is always 0 for short inferences.
 *   BATTERY_PROPERTY_CURRENT_NOW gives µA-level resolution, updated in real-time.
 *   We sample before + after inference, average them, and multiply by duration.
 *   This gives a proxy proportional to actual energy consumed (valid for Pareto).
 */
class DeviceMonitor {

    /** Battery level as percentage (0-100). Coarse, for UI display. */
    fun getBatteryLevel(context: Context): Int {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        return bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
    }

    /**
     * Instantaneous battery current in microamps (µA).
     * Positive = discharging, negative = charging (device-dependent).
     * Returns absolute value for consistency.
     * Returns 0 if sensor is unavailable.
     */
    fun getBatteryMicroAmps(context: Context): Long {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val current = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        return kotlin.math.abs(current)
    }

    /**
     * Compute energy proxy from current samples and duration.
     *
     * @param microAmpsBefore  µA sample taken before inference
     * @param microAmpsAfter   µA sample taken after inference
     * @param durationMs       inference wall-clock time in ms
     * @return energy proxy in µA·ms (proportional to joules)
     */
    fun computeEnergyProxy(microAmpsBefore: Long, microAmpsAfter: Long, durationMs: Long): Long {
        val avgCurrent = (microAmpsBefore + microAmpsAfter) / 2
        return avgCurrent * durationMs
    }

    /** Memory usage as percentage of total RAM. */
    fun getMemoryUsage(context: Context): Float {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)

        val totalMem = memoryInfo.totalMem.toFloat()
        val availMem = memoryInfo.availMem.toFloat()

        return if (totalMem > 0) {
            ((totalMem - availMem) / totalMem) * 100f
        } else {
            0f
        }
    }
}
