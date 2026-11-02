using UnityEngine;

namespace ScoutsanWholeBody
{

    [System.Serializable]
    public class TargetCurriculum
    {
        [Tooltip("Close start radius (m), straight ahead of the robot.")]
        public float minRadius = 0.75f;
        [Tooltip("End of the straight-line distance phase (m); the arc begins here and the far distance stays this.")]
        public float lineRadius = 2.0f;
        [Tooltip("Academy steps to grow minRadius -> lineRadius straight ahead (phase 1).")]
        public int distanceFullSteps = 2000000;
        [Tooltip("TOTAL arc angle (deg): the target spawns within +-(this/2) of straight-ahead. 40 = +-20 per side.")]
        public float arcTotalDeg = 40f;
        [Tooltip("Academy steps to grow the arc 0 -> +-(arcTotalDeg/2), AFTER the distance phase (phase 2).")]
        public int arcFullSteps = 2000000;
        [Tooltip("Target height above the robot base (m).")]
        public float height = 0.25f;
        [Tooltip("Target frame convention (confondu). ON (default) = +Z points AWAY from the robot (radially outward, base->target), +Y up, +X right -- matches the gripper's approaching +Z so the frames can coincide. OFF = legacy 'face the robot' (+Z points back at the base).")]
        public bool zAwayFromRobot = true;

        public void Sample(Vector3 basePos, Quaternion baseRot, long totalSteps, out Vector3 pos, out Quaternion rot)
        {
            float dp = (distanceFullSteps > 0) ? Mathf.Clamp01(totalSteps / (float)distanceFullSteps) : 1f;
            float radius = Mathf.Lerp(minRadius, lineRadius, dp);
            float ap = Mathf.Clamp01((totalSteps - distanceFullSteps) / (float)Mathf.Max(1, arcFullSteps));
            float az = Random.Range(-1f, 1f) * (arcTotalDeg * 0.5f) * ap * Mathf.Deg2Rad;
            Compose(basePos, baseRot, radius, az, out pos, out rot);
        }

        public void SampleFullRange(Vector3 basePos, Quaternion baseRot, out Vector3 pos, out Quaternion rot)
        {
            long span = (long)distanceFullSteps + (long)Mathf.Max(1, arcFullSteps);
            long steps = (long)(Random.value * span);
            Sample(basePos, baseRot, steps, out pos, out rot);
        }

        public void SampleArcBiased(Vector3 basePos, Quaternion baseRot, float lineFraction, out Vector3 pos, out Quaternion rot)
        {
            if (Random.value < Mathf.Clamp01(lineFraction))
            {
                float radius = Random.Range(minRadius, lineRadius);
                Compose(basePos, baseRot, radius, 0f, out pos, out rot);
            }
            else
            {
                float az = Random.Range(-1f, 1f) * (arcTotalDeg * 0.5f) * Mathf.Deg2Rad;
                Compose(basePos, baseRot, lineRadius, az, out pos, out rot);
            }
        }

        public void SampleLineArc(Vector3 basePos, Quaternion baseRot, int index, int total, int lineN, out Vector3 pos, out Quaternion rot)
        {
            total = Mathf.Max(2, total);
            lineN = Mathf.Clamp(lineN, 1, total - 1);
            int arcN  = total - lineN;
            index = ((index % total) + total) % total;
            if (index < lineN)
            {
                float r01 = (lineN == 1) ? 1f : index / (float)(lineN - 1);
                Compose01(basePos, baseRot, r01, 0f, out pos, out rot);
            }
            else
            {
                int k = index - lineN;
                float a01 = (arcN == 1) ? 0f : Mathf.Lerp(-1f, 1f, k / (float)(arcN - 1));
                Compose01(basePos, baseRot, 1f, a01, out pos, out rot);
            }
        }

        public void Compose01(Vector3 basePos, Quaternion baseRot, float radius01, float az01, out Vector3 pos, out Quaternion rot)
        {
            float radius = Mathf.Lerp(minRadius, lineRadius, Mathf.Clamp01(radius01));
            float az = Mathf.Clamp(az01, -1f, 1f) * (arcTotalDeg * 0.5f) * Mathf.Deg2Rad;
            Compose(basePos, baseRot, radius, az, out pos, out rot);
        }

        void Compose(Vector3 basePos, Quaternion baseRot, float radius, float azRad, out Vector3 pos, out Quaternion rot)
        {
            Vector3 fwd   = Vector3.ProjectOnPlane(baseRot * Vector3.forward, Vector3.up).normalized;
            Vector3 right = Vector3.Cross(Vector3.up, fwd);
            Vector3 dir = Mathf.Cos(azRad) * fwd + Mathf.Sin(azRad) * right;
            pos = basePos + radius * dir;
            pos.y = basePos.y + height;
            Vector3 facing = zAwayFromRobot ? dir : -dir;
            rot = Quaternion.LookRotation(facing, Vector3.up);
        }
    }
}
