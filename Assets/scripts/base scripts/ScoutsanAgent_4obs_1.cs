using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class ScoutsanAgent_4obs_1 : Agent
{
    [Header("Wheels (drag from Hierarchy)")]
    public ArticulationBody frontLeftWheel;
    public ArticulationBody frontRightWheel;
    public ArticulationBody rearLeftWheel;
    public ArticulationBody rearRightWheel;

    [Header("Target sphere")]
    public Transform target;

    public float wheelRadius = 0.16459f;
    public float wheelSeparation = 0.58306f;

    [Header("Action limits (match real Scout V2 specs)")]
    public float maxLinearVel = 0.5f;
    public float maxAngularVel = 0.8f;

    [Header("Wheel drive tuning")]
    public float wheelForceLimit = 100f;
    public float wheelDamping = 5000f;

    [Header("Action smoothing")]
    public float maxVAccel = 0.5f;
    public float maxWAccel = 0.8f;

    [Header("Task config")]
    public float arenaHalfSize = 5.0f;
    public float minSpawnDistance = 1.5f;
    public float successRadius = 0.6f;
    public float fallY = -1.0f;

    [Header("Reward weights")]
    public float progressCoeff = 0.1f;
    public float timePenalty   = 0.001f;
    public float bearingCoeff  = 0.005f;

    [Header("Bearing fade (prevents hunting near goal)")]
    public float bearingFadeStart = 1.8f;
    public float bearingFadeEnd   = 0.8f;

    ArticulationBody baseBody;
    Vector3 startPosition;
    Quaternion startRotation;
    float prevDistance;
    float prevV = 0f;
    float prevW = 0f;

    public override void Initialize()
    {
        baseBody = GetComponent<ArticulationBody>();
        startPosition = transform.position;
        startRotation = transform.rotation;
        Time.timeScale = 1f;
    }

    public override void OnEpisodeBegin()
    {
        if (baseBody != null)
        {
            baseBody.TeleportRoot(startPosition, startRotation);
            baseBody.velocity = Vector3.zero;
            baseBody.angularVelocity = Vector3.zero;
        }

        SetWheelVelocity(frontLeftWheel, 0f);
        SetWheelVelocity(frontRightWheel, 0f);
        SetWheelVelocity(rearLeftWheel, 0f);
        SetWheelVelocity(rearRightWheel, 0f);

        prevV = 0f;
        prevW = 0f;

        Vector3 newTarget;
        int safety = 0;
        do
        {
            newTarget = new Vector3(
                startPosition.x + Random.Range(-arenaHalfSize, arenaHalfSize),
                target.position.y,
                startPosition.z + Random.Range(-arenaHalfSize, arenaHalfSize)
            );
            safety++;
        }
        while (Vector3.Distance(newTarget, startPosition) < minSpawnDistance && safety < 20);
        target.position = newTarget;

        prevDistance = Vector3.Distance(transform.position, target.position);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Vector3 toTargetWorld = target.position - transform.position;
        Vector3 toTargetLocal = transform.InverseTransformDirection(toTargetWorld);

        float forwardDist = toTargetLocal.z;
        float lateralDist = -toTargetLocal.x;

        Vector3 linVelLocal = transform.InverseTransformDirection(baseBody.velocity);
        float linVelForward = linVelLocal.z;

        float angVel        = -baseBody.angularVelocity.y;

        sensor.AddObservation(forwardDist);
        sensor.AddObservation(lateralDist);
        sensor.AddObservation(linVelForward);
        sensor.AddObservation(angVel);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {

        float vRos = actionBuffers.ContinuousActions[0];
        float wRos = actionBuffers.ContinuousActions[1];

        float v =  vRos;
        float w = -wRos;

        float halfL  = wheelSeparation * 0.5f;
        float vLeft  = (v - w * halfL) / wheelRadius;
        float vRight = (v + w * halfL) / wheelRadius;

        SetWheelVelocity(frontLeftWheel,  -vLeft);
        SetWheelVelocity(rearLeftWheel,   -vLeft);
        SetWheelVelocity(frontRightWheel,  vRight);
        SetWheelVelocity(rearRightWheel,   vRight);

        float distance = Vector3.Distance(transform.position, target.position);
        float progress = prevDistance - distance;
        AddReward(progressCoeff * progress);
        prevDistance = distance;

        AddReward(-timePenalty);

        Vector3 toTargetWorld = target.position - transform.position;
        Vector3 toTargetLocalCheck = transform.InverseTransformDirection(toTargetWorld);
        Vector3 toTargetDir = toTargetWorld.normalized;

        float bearing;
        if (toTargetLocalCheck.z >= 0f)
        {

            bearing = Vector3.Angle(transform.forward, toTargetDir) * Mathf.Deg2Rad;
        }
        else
        {

            bearing = Vector3.Angle(-transform.forward, toTargetDir) * Mathf.Deg2Rad;
        }

        float bearingFade = Mathf.Clamp01(
            (distance - bearingFadeEnd) / (bearingFadeStart - bearingFadeEnd)
        );

        AddReward(-bearingCoeff * bearing * bearingFade);

        if (distance < successRadius)
        {
            SetReward(1.0f);
            EndEpisode();
            return;
        }
        if (transform.position.y < fallY)
        {
            SetReward(-1.0f);
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {

        var ca = actionsOut.ContinuousActions;
        ca[0] =  Input.GetAxis("Vertical");
        ca[1] = -Input.GetAxis("Horizontal");
    }

    void SetWheelVelocity(ArticulationBody wheel, float radPerSec)
    {
        if (wheel == null) return;
        var drive = wheel.xDrive;
        drive.targetVelocity = radPerSec * Mathf.Rad2Deg;
        drive.forceLimit = wheelForceLimit;
        drive.damping = wheelDamping;
        drive.stiffness = 0f;
        wheel.xDrive = drive;
    }
}
