using System.IO;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using ScoutsanWholeBody;

public class ScoutsanV1_1b : Agent
{

    public ArticulationBody frontLeftWheel;
    public ArticulationBody frontRightWheel;
    public ArticulationBody rearLeftWheel;
    public ArticulationBody rearRightWheel;

    const float wheelRadius = 0.16459f;
    const float wheelSeparation = 0.58306f;
    const float wheelForceLimit = 100f;
    const float wheelDamping = 5000f;

    [Tooltip("Wheel commanded-velocity cap (deg/s), ported VERBATIM from the wb Path B agents. SetWheelVelocity clamps each wheel's targetVelocity to +-this so the command is achievable on the grippy robot. 150 = wb1am (base reaches ~0.30 m/s); 70 = wb_f (chokes the base to ~0.20 m/s). Off-switch for the wheel-cap A/B: raise it high to disable. Requires a REBUILD to bake.")]
    public float maxWheelVelocityDeg = 150f;

    const float maxLinearVel = 0.3f;
    const float maxAngularVel = 0.4f;

    public ArticulationBody[] armJoints = new ArticulationBody[6];

    public Transform fingerLink1;
    public Transform fingerLink2;

    public Vector3 gripperFrameEuler = Vector3.zero;

    public Vector3 gripperTipOffset = new Vector3(0f, 0.02f, 0f);

    public Transform target;

    [Tooltip("FIXED per-episode target offset from THIS robot's ROOT start position (startPosition), WORLD axes (x, y-up, z-forward). Every arena is a duplicate, so the SAME offset lands the target at the same spot in each arena & episode. Default (0, 0.5, 3) = target_local(0,0.9,3) - root_local(0,0.4,0).")]
    public Vector3 targetSpawnOffset = new Vector3(0f, 0.5f, 3f);

    [Tooltip("s8b: randomize the target POSITION each episode inside a front patch around targetSpawnOffset. false = fixed target (SampleScene / s8 behaviour). Orientation is NEVER randomized.")]
    public bool randomizeTargetPosition = false;

    [Header("Eval samples (comparison harness): grid sweep; set the SAME grid here and in the Path B scene")]
    [Tooltip("ON = spawn the target on a deterministic (x,y,z) grid, ONE PER EPISODE, cycling -- for the fair comparison eval. Leave OFF for training. Ported verbatim from ScoutsanWholeBodyAgent so both agents sweep the identical set.")]
    public bool useEvalSamples = false;
    const float evalZMin = 0.75f, evalZMax = 3.0f;   const int evalZSteps = 4;
    const float evalXMin = -1.0f, evalXMax = 1.0f;   const int evalXSteps = 3;
    const float evalYMin = 0.25f, evalYMax = 0.5f;   const int evalYSteps = 2;
    int evalSampleIdx = 0;
    const float spawnRangeX = 2f;
    const float spawnRangeZ = 1.5f;
    [Tooltip("If true, grow the spawn half-ranges from spawnCurriculumStartFrac to 1.0 over spawnCurriculumFullSteps Academy steps (protects the warm start). false = full range from step 0.")]
    public bool useSpawnCurriculum = true;
    const float spawnCurriculumStartFrac = 0.15f;
    const int spawnCurriculumFullSteps = 3000000;

    const float minSpawnDistance = 1.5f;
    const int maxSpawnAttempts = 20;

    [Tooltip("Randomize the target ORIENTATION each episode = YAW about the vertical axis by +-orientTiltRangeDeg (the X and Z axes rotate around Y; Y stays up). false = zero yaw (upright, fixed heading). The support box yaws with it.")]
    public bool randomizeTargetOrientation = false;
    [Tooltip("Half-range (deg) of the target YAW about the vertical (Y) axis at FULL curriculum. X/Z rotate around Y; Y stays up. Default 20. (Name is historical: this is a yaw range, not a tilt.)")]
    const float orientTiltRangeDeg = 20f;
    [Tooltip("If true, grow the tilt half-range from orientCurriculumStartFrac to 1.0 over orientCurriculumFullSteps Academy steps. false = full tilt from step 0.")]
    public bool useOrientCurriculum = true;
    [Tooltip("Orientation curriculum start fraction (0.0 = start from ZERO tilt = where the s8b warm start is competent).")]
    const float orientCurriculumStartFrac = 0.0f;
    [Tooltip("Academy TotalStepCount at which the tilt reaches +-orientTiltRangeDeg. Linear ramp from start-frac to 1.0.")]
    const int orientCurriculumFullSteps = 3000000;

    public Transform baseLink;

    const int hardSnapSteps = 3;

    static readonly int[] controlledJointIndices = { 0, 1, 2, 3, 4, 5 };

    int controlledJointCount => controlledJointIndices != null ? controlledJointIndices.Length : 0;

    public bool traceJoints = true;
    const int traceEveryNSteps = 20;

    const float maxJointDeltaDeg = 3f;

    public bool  adaptiveActionScale = true;
    const float fineDistance  = 0.20f;
    const float fineOrient    = 0.30f;
    const float fineDeltaFrac = 0.03f;

    static readonly float[] jointMinDeg  = { -360f, -95f, -135f, -360f, -135f, -360f };
    static readonly float[] jointMaxDeg  = {  360f,  95f,  135f,  360f,  135f,  360f };

    [Tooltip("Episode-reset arm pose in degrees, J1..J6. {0,0,-90,0,0,0} = L shape (matches the wb Path B agent); " +
             "{0,0,0,0,0,0} = the legacy straight-up pose. Must have exactly 6 entries.")]
    public float[] jointHomeDeg = { 0f, 0f, -90f, 0f, 0f, 0f };
    static readonly float[] jointHomeFallbackDeg = { 0f, 0f, -90f, 0f, 0f, 0f };

    float JointHomeDeg(int i) =>
        (jointHomeDeg != null && jointHomeDeg.Length == 6) ? jointHomeDeg[i] : jointHomeFallbackDeg[i];

    const float armStiffness  = 10000f;
    const float armDamping    = 200f;
    const float armForceLimit = 1000f;
    public float maxJointVelocityDeg = 30f;

    [Header("Success thresholds (tune in the Inspector)")]
    [Tooltip("Position success radius (m): gripper must be within this of the target")]
    public float successRadius = 0.03f;

    const float distanceRewardScale = 10.0f;
    const float progressClamp       = 0.25f;
    const float reachReward         = 100.0f;
    const float directionalReward   = 0.01f;
    const float orientDirectionalReward = 0.01f;
    [Tooltip("Inner reward-band radius (m)")]
    const float regionInnerRadius   = 0.05f;
    const float regionInnerReward   = 1.0f;
    [Tooltip("Outer reward-band radius (m)")]
    const float regionOuterRadius   = 0.10f;
    const float regionOuterReward   = 0.5f;
    [Tooltip("Mid-field milestone band (m): one-shot bonus when d first drops below this mid-way through the drive-in")]
    const float regionMidRadius     = 1.0f;
    const float regionMidReward     = 0.5f;
    [Tooltip("Far-field milestone band (m): one-shot bonus for the first big chunk of the 3 m approach")]
    const float regionFarRadius     = 2.0f;
    const float regionFarReward     = 0.3f;
    const float orientRewardScale  = 0.5f;
    [Tooltip("Orientation success tolerance (rad)")]
    public float orientToleranceRad = 0.05f;

    [Header("Orientation milestone band (confondu)")]
    [Tooltip("Inner orientation band (rad): crossing into it pays orientInnerReward once per episode")]
    const float orientInnerToleranceRad = 0.08f;
    const float orientInnerReward       = 8.0f;
    [Tooltip("Outer orientation band (rad): crossing into it pays orientOuterReward once per episode")]
    const float orientOuterToleranceRad = 0.25f;
    const float orientOuterReward       = 3.0f;

    [Header("Trajectory smoothing (control-action penalty)")]
    [Tooltip("w_ctrl: weight on sum of squared actions, subtracted from the reward each step (0 = off)")]
    const float controlActionRewardScale = 0.05f;

    [Tooltip("w_rate: weight on sum of squared action DIFFERENCES (a_t - a_{t-1})^2, subtracted each step (0 = off)")]
    const float actionRatePenaltyScale = 0.05f;

    public Vector3 targetFrameOffsetEuler = Vector3.zero;

    [Tooltip("Use the WDLS T-arc target curriculum (on-axis line + shallow arc, WORLD axes) INSTEAD of the rectangular patch. ON => randomizeTargetPosition/Orientation are ignored. The on-axis center reproduces the s9_np fixed target (identity rotation).")]
    public bool useTArcCurriculum = false;
    [Tooltip("WATCH ONLY: sample the FULL T (line+arc) every episode instead of the step-ramp. Turn ON when watching a converged model in the Editor -- Academy step count resets to 0 on Play, so the ramp would otherwise show only the narrow curriculum start. Leave OFF for training.")]
    public bool tArcFullRange = false;
    [Tooltip("WATCH ONLY: cycle the SAME deterministic line+arc target set wb_1am was evaluated on (SampleLineArc), one per episode -- for a FAIR s9-vs-wb1am comparison. Overrides tArcFullRange. wb_1am used count 5, lineN 2.")]
    public bool tArcLineArc = false;
    public int  tArcLineArcCount = 5;
    public int  tArcLineArcLineCount = 2;
    int         tArcEpisodeIdx = 0;
    [Tooltip("The T-arc curriculum (line + arc). FAST warm-start defaults: distanceFullSteps 500k + arcFullSteps 1M -> full T by ~1.5M Academy steps. minRadius/lineRadius/arcTotalDeg/height are the WDLS values (0.75 / 2 / 40 / 0.25).")]
    public TargetCurriculum tArc = new TargetCurriculum { distanceFullSteps = 500000, arcFullSteps = 1000000 };

    [Tooltip("Per-episode target yaw (deg) about the vertical axis (world Y = your z-up). alpha ~ U(-range,+range); z stays up, only the heading varies. Narrow start = 20 deg. SERIALIZED in the scene -> Inspector value governs the build.")]
    const float targetYawRangeDeg = 20f;
    Quaternion targetBaseRotation = Quaternion.identity;

    public bool drawFrameGizmos = true;

    const float gizmoAxisLength = 0.3f;

    const float gizmoAxisThickness = 6f;

    const float floorY = 0f;

    const float collisionPenalty = 50.0f;

    [Tooltip("Support box (table/drawer) under the target sphere. Assign the box object carrying the ArticulationBody (e.g. scoutsan_box); its articulation ROOT is resolved at runtime and teleported to track the sphere each episode. EMPTY = box tracking disabled (box-free, like SampleScene).")]
    public Transform supportBox;
    [Tooltip("Pin the box immovable at startup so, once teleported, it stays put (a movable root would fall or get shoved). Off-switch for A/B.")]
    public bool makeBoxImmovable = true;
    [Tooltip("If true, an arm part touching the SUPPORT BOX costs a PER-STEP penalty instead of ending the episode -- so the arm can reach the object resting on the box and still earn the +100 confondu. Self/floor/other-environment collisions STILL end the episode. false = box contact is fatal (the transfer-proven default).")]
    public bool softBoxCollision = false;
    [Tooltip("Per-step penalty while ANY arm part overlaps the support box (only when softBoxCollision=true). Applied ONCE per step. 1.0 -> a ~15-step brush costs ~-15, vs +100 confondu and -50 fatal.")]
    const float boxTouchPenalty = 1.0f;
    [Tooltip("If true, the BASE (base_link) or any of the 4 WHEELS touching the support box (the 'Cube') is a FATAL collision -> -collisionPenalty and the episode restarts, so the base cannot ram the cube the target sits on. The ARM keeps its own checks above. Only the box counts (the floor never trips it). false = base/wheel-vs-box ignored (old behaviour). Off-switch for A/B.")]
    public bool baseBoxCollisionFatal = true;

    static readonly string[] collisionExclusions = { "scoutsan_box_1" };

    static readonly string[] gripperSelfCollisionGroup = { "base_link_0", "finger1_Link_0", "finger2_Link_0" };

    [Header("Fall / tilt detection (base)")]
    [Tooltip("Chassis height (m) below which the base counts as fallen -> FAIL + restart")]
    const float fallY = -0.5f;
    [Tooltip("Chassis tilt (deg from upright) above which the base counts as fallen -> FAIL + restart")]
    const float maxTiltDeg = 50f;
    [Tooltip("One-off penalty applied when the base falls, just before the episode restarts")]
    const float fallPenalty = 50f;

    ArticulationBody baseBody;
    Vector3 startPosition;
    Quaternion startRotation;

    ArticulationBody boxBody;
    Vector3 boxRootOffset;
    Quaternion boxRootRot;
    bool boxReady = false;
    readonly HashSet<Collider> boxColliderSet = new HashSet<Collider>();
    float boxTouchTotal = 0f;
    int   boxTouchSteps = 0;

    float[] armTargets = new float[6];

    const int NumActions = 8;
    readonly float[] prevActions = new float[8];
    bool havePrevActions = false;
    bool ready = false;
    int snapStepsLeft = 0;
    bool collisionHandled = false;

    bool baselineEstablished = false;
    bool baselinePending = false;

    bool actionMismatchLogged = false;
    bool firstActionLogged = false;

    float prevDistance = 0f;

    float prevPhi = 0f;
    bool innerBandAwarded = false;
    bool outerBandAwarded = false;
    bool midBandAwarded = false;
    bool farBandAwarded = false;
    bool innerOrientAwarded = false;
    bool outerOrientAwarded = false;

    readonly List<Collider> armColliders = new List<Collider>();
    readonly HashSet<Collider> armColliderSet = new HashSet<Collider>();
    readonly HashSet<(int, int)> baselineSelfPairs = new HashSet<(int, int)>();
    readonly HashSet<(int, int)> baselineEnvPairs = new HashSet<(int, int)>();
    readonly Collider[] overlapBuffer = new Collider[32];

    readonly List<Collider> baseColliders = new List<Collider>();
    readonly HashSet<Collider> baseColliderSet = new HashSet<Collider>();

    StreamWriter logWriter;
    int episodeCount = 0;
    string lastEndReason = "n/a";

    float epR1, epR3, epR4, epR5, epR6, epR8, epTerm;
    int epSteps;
    static System.IO.StreamWriter s_rewardCsv;
    static bool s_rewardCsvTried;
    static readonly object s_rewardCsvLock = new object();

    public override void Initialize()
    {

        baseBody = ResolveArticulationRoot();
        if(baseBody == null)
        {
            startPosition = transform.position;
            startRotation = transform.rotation;
        }
        else
        {
            startPosition = baseBody.transform.position;
            startRotation = baseBody.transform.rotation;
        }

        string logDir = Path.Combine(Application.dataPath, "..", "Logs");
        Directory.CreateDirectory(logDir);

        string logPath = Path.Combine(logDir, $"arm_agent_l_{System.DateTime.Now:yyyyMMdd_HHmmss}_{GetInstanceID()}.log");
        logWriter = new StreamWriter(logPath, append: false) { AutoFlush = true };
        Log($"=== training log started: {logPath} ===");

        targetBaseRotation = (target != null) ? target.rotation : Quaternion.identity;

        if (supportBox != null && target != null)
        {
            boxBody = FindArticulationRootOf(supportBox);
            Transform boxT = (boxBody != null) ? boxBody.transform : supportBox;

            boxRootOffset  = Quaternion.Inverse(targetBaseRotation) * (boxT.position - target.position);
            boxRootRot     = Quaternion.Inverse(targetBaseRotation) * boxT.rotation;
            if (boxBody != null && makeBoxImmovable) boxBody.immovable = true;
            boxColliderSet.Clear();
            foreach (var c in boxT.GetComponentsInChildren<Collider>(true)) boxColliderSet.Add(c);
            boxReady = true;
            Log($"support box ready: root='{(boxBody != null ? boxBody.name : supportBox.name)}' offset={boxRootOffset} colliders={boxColliderSet.Count} soft={softBoxCollision} penalty={boxTouchPenalty}");
        }
        else
        {
            Log("support box: none assigned (supportBox slot empty) -- box tracking disabled (box-free).");
        }
    }

          ArticulationBody ResolveArticulationRoot()
    {
        ArticulationBody ab = GetComponent<ArticulationBody>();
        if (ab == null && baseLink != null) ab = baseLink.GetComponent<ArticulationBody>();
        if (ab == null && baseLink != null) ab = baseLink.GetComponentInParent<ArticulationBody>();
        if (ab == null) ab = GetComponentInChildren<ArticulationBody>();
        if (ab == null) return null;

        ArticulationBody root = ab;
        while (root.transform.parent != null)
        {
            ArticulationBody parent = root.transform.parent.GetComponentInParent<ArticulationBody>();
            if (parent == null || parent == root) break;
            root = parent;
        }
        return root;
    }

    ArticulationBody FindArticulationRootOf(Transform t)
    {
        if (t == null) return null;
        ArticulationBody ab = t.GetComponent<ArticulationBody>();
        if (ab == null) ab = t.GetComponentInParent<ArticulationBody>();
        if (ab == null) ab = t.GetComponentInChildren<ArticulationBody>();
        if (ab == null) return null;

        ArticulationBody root = ab;
        while (root.transform.parent != null)
        {
            ArticulationBody parent = root.transform.parent.GetComponentInParent<ArticulationBody>();
            if (parent == null || parent == root) break;
            root = parent;
        }
        return root;
    }

    void PlaceSupportBox()
    {
        if (!boxReady || supportBox == null) return;

        Vector3 pos    = target.position + target.rotation * boxRootOffset;
        Quaternion rot = target.rotation * boxRootRot;
        if (boxBody != null)
        {
            boxBody.TeleportRoot(pos, rot);
            boxBody.velocity = Vector3.zero;
            boxBody.angularVelocity = Vector3.zero;
        }
        else
        {
            supportBox.SetPositionAndRotation(pos, rot);
        }
    }

    void OnDestroy()
    {
        Log("=== training log ended ===");
        logWriter?.Flush();
        logWriter?.Close();
        logWriter = null;
    }

    void Log(string message)
    {
        Debug.Log($"[ScoutsanV1_1b] {message}");
        logWriter?.WriteLine($"[{System.DateTime.Now:HH:mm:ss.fff}] {message}");
    }

    Matrix4x4 WorldToFrameMatrix(Transform frame)
    {
        return Matrix4x4.TRS(frame.position, frame.rotation, Vector3.one).inverse;
    }

    Vector3 WorldToLocalPoint(Transform frame, Vector3 worldPoint)
    {
        return WorldToFrameMatrix(frame).MultiplyPoint3x4(worldPoint);
    }

    Quaternion LocalRotation(Quaternion worldRot)
    {
        Quaternion q = Quaternion.Inverse(baseLink.rotation) * worldRot;
        if (q.w < 0f) { q.x = -q.x; q.y = -q.y; q.z = -q.z; q.w = -q.w; }
        return q;
    }

    bool HasFingers => fingerLink1 != null && fingerLink2 != null;

    Vector3 GripperPosition()
    {
        Vector3 mid = HasFingers ? 0.5f * (fingerLink1.position + fingerLink2.position)
                    : (fingerLink1 != null ? fingerLink1.position : transform.position);
        return mid + GripperRotation() * gripperTipOffset;
    }

    Quaternion GripperRotation()
    {
        Quaternion baseRot = fingerLink1 != null ? fingerLink1.rotation
                           : (fingerLink2 != null ? fingerLink2.rotation : transform.rotation);
        return baseRot * Quaternion.Euler(gripperFrameEuler);
    }

    ArticulationBody RootOf(ArticulationBody body)
    {
        ArticulationBody cur = body;
        while (cur != null && !cur.isRoot)
        {
            Transform p = cur.transform.parent;
            cur = (p != null) ? p.GetComponentInParent<ArticulationBody>() : null;
        }
        return cur;
    }

    void EnsureReady()
    {
        if (ready) return;

        if (armJoints[0] == null) { Debug.LogError("ArmAgentL: armJoints[0] is null."); return; }

        float maxJointVelocityRad = maxJointVelocityDeg * Mathf.Deg2Rad;
        foreach (var j in armJoints)
            if (j != null) j.maxJointVelocity = maxJointVelocityRad;

        for (int i = 0; i < armJoints.Length; i++)
        {
            var j = armJoints[i];
            if (j == null) { Log($"joint setup J{i + 1}: armJoints[{i}] is null."); continue; }

            j.jointFriction = 0f;
            j.angularDamping = 0f;

            if (j.twistLock == ArticulationDofLock.LockedMotion)
                j.twistLock = ArticulationDofLock.LimitedMotion;

            var drive = j.xDrive;

            drive.lowerLimit = jointMinDeg[i];
            drive.upperLimit = jointMaxDeg[i];

            j.xDrive = drive;

            Log($"joint setup J{i + 1} '{j.name}': dof={j.dofCount}, jointType={j.jointType}, " +
                $"twistLock={j.twistLock}, xDrive limits=[{j.xDrive.lowerLimit:F1},{j.xDrive.upperLimit:F1}], " +
                $"stiffness={j.xDrive.stiffness:F0}, forceLimit={j.xDrive.forceLimit:F0}, " +
                $"friction={j.jointFriction:F1}, mass={j.mass:F2}");
        }

        CollectArmColliders();
        CollectBaseColliders();

        ready = true;
    }

    bool IsExcludedFromCollision(Collider c)
    {
        if (target != null && c.transform == target) return true;
        foreach (var name in collisionExclusions)
            if (c.name == name) return true;
        return false;
    }

    bool IsGripperPart(Collider c)

    {
        foreach (var name in gripperSelfCollisionGroup)
            if (c.name == name) return true;
        return false;
    }

    void CollectArmColliders()
    {
        foreach (var body in armJoints)
        {
            if (body == null) continue;
            foreach (var col in body.GetComponentsInChildren<Collider>(true))
            {
                if (armColliderSet.Add(col))
                    armColliders.Add(col);
            }
        }
        Log($"CollectArmColliders: found {armColliders.Count} colliders.");
    }

    void CollectBaseColliders()
    {
        baseColliders.Clear(); baseColliderSet.Clear();
        if (baseLink != null)
            foreach (var c in baseLink.GetComponentsInChildren<Collider>(true))
                if (!armColliderSet.Contains(c) && baseColliderSet.Add(c)) baseColliders.Add(c);
        foreach (var wheel in new[] { frontLeftWheel, frontRightWheel, rearLeftWheel, rearRightWheel })
            if (wheel != null)
                foreach (var c in wheel.GetComponentsInChildren<Collider>(true))
                    if (!armColliderSet.Contains(c) && baseColliderSet.Add(c)) baseColliders.Add(c);
        Log($"CollectBaseColliders: {baseColliders.Count} base/wheel colliders (fatal on cube = {baseBoxCollisionFatal}).");
    }

    void CheckBaseBoxCollision()
    {
        if (boxColliderSet.Count == 0) return;
        for (int i = 0; i < baseColliders.Count; i++)
        {
            var col = baseColliders[i];
            if (col == null) continue;
            int hitCount = Physics.OverlapBoxNonAlloc(col.bounds.center, col.bounds.extents, overlapBuffer, Quaternion.identity);
            for (int h = 0; h < hitCount; h++)
            {
                var hit = overlapBuffer[h];
                if (hit == col || hit.isTrigger || !boxColliderSet.Contains(hit)) continue;
                bool overlapping = Physics.ComputePenetration(
                    col, col.transform.position, col.transform.rotation,
                    hit, hit.transform.position, hit.transform.rotation, out _, out float dist);
                if (overlapping && dist > 0.0001f)
                {
                    FailEpisode($"base '{col.name}' hit the cube '{hit.name}'");
                    return;
                }
            }
        }
    }

    void EstablishCollisionBaseline()

    {
        int envPairs = 0, selfPairs = 0;

        for (int i = 0; i < armColliders.Count; i++)
        {
            var col = armColliders[i];

            int hitCount = Physics.OverlapBoxNonAlloc(col.bounds.center, col.bounds.extents, overlapBuffer, Quaternion.identity);
            for (int h = 0; h < hitCount; h++)
            {
                var hit = overlapBuffer[h];
                if (hit == col || hit.isTrigger) continue;
                if (hit is MeshCollider hitMesh && !hitMesh.convex) continue;
                if (IsExcludedFromCollision(hit)) continue;

                bool overlapping = Physics.ComputePenetration(
                    col, col.transform.position, col.transform.rotation,
                    hit, hit.transform.position, hit.transform.rotation,
                    out _, out float dist);

                if (!overlapping || dist <= 0.0001f) continue;

                Physics.IgnoreCollision(col, hit, true);

                if (armColliderSet.Contains(hit))
                {
                    int j = armColliders.IndexOf(hit);
                    if (baselineSelfPairs.Add((Mathf.Min(i, j), Mathf.Max(i, j))))
                        selfPairs++;
                }
                else
                {
                    if (baselineEnvPairs.Add((i, hit.GetInstanceID())))
                        envPairs++;
                }
            }
        }
        Log($"EstablishCollisionBaseline: {envPairs} structural environment pairs, {selfPairs} structural self-touch pairs ignored at rest.");
    }

    void CheckSelfCollisions()

    {
        for (int i = 0; i < armColliders.Count; i++)
        {
            for (int j = i + 1; j < armColliders.Count; j++)
            {
                if (baselineSelfPairs.Contains((i, j))) continue;

                var a = armColliders[i];
                var b = armColliders[j];

                if (IsGripperPart(a) && IsGripperPart(b)) continue;

                bool overlapping = Physics.ComputePenetration(
                    a, a.transform.position, a.transform.rotation,
                    b, b.transform.position, b.transform.rotation,
                    out _, out float dist);

                if (overlapping && dist > 0.0001f)
                {
                    FailEpisode($"self-collision '{a.name}' <-> '{b.name}'");
                    return;
                }
            }
        }
    }

    void CheckEnvironmentCollisions()

    {
        bool boxTouched = false;
        for (int i = 0; i < armColliders.Count; i++)
        {
            var col = armColliders[i];

            if (col.bounds.min.y < floorY - 0.0005f)
            {
                FailEpisode($"'{col.name}' hit the floor (y={col.bounds.min.y:F4})");
                return;
            }

            int hitCount = Physics.OverlapBoxNonAlloc(col.bounds.center, col.bounds.extents, overlapBuffer, Quaternion.identity);
            for (int h = 0; h < hitCount; h++)
            {
                var hit = overlapBuffer[h];
                if (hit == col || hit.isTrigger) continue;
                if (armColliderSet.Contains(hit)) continue;
                if (hit is MeshCollider hitMesh && !hitMesh.convex) continue;
                if (baselineEnvPairs.Contains((i, hit.GetInstanceID()))) continue;
                if (IsExcludedFromCollision(hit)) continue;

                bool overlapping = Physics.ComputePenetration(
                    col, col.transform.position, col.transform.rotation,
                    hit, hit.transform.position, hit.transform.rotation,
                    out _, out float dist);

                if (overlapping && dist > 0.0001f)
                {

                    if (softBoxCollision && boxColliderSet.Contains(hit))
                    {
                        boxTouched = true;
                        continue;
                    }
                    FailEpisode($"'{col.name}' hit environment '{hit.name}'");
                    return;
                }
            }
        }

        if (boxTouched)
        {
            AddReward(-boxTouchPenalty);
            boxTouchTotal += boxTouchPenalty;
            boxTouchSteps++;
        }
    }

    void FailEpisode(string reason)

    {
        if (collisionHandled) return;
        collisionHandled = true;
        lastEndReason = $"FAIL: {reason}";
        Log($"Collision failure: {reason} -- penalty {-collisionPenalty}, ending episode.");
        AddReward(-collisionPenalty);
        epTerm = -collisionPenalty;
        EndEpisode();
    }

    void EnsureRewardCsv()
    {
        if (s_rewardCsvTried) return;
        s_rewardCsvTried = true;
        string path = System.Environment.GetEnvironmentVariable("SCOUTSAN_REWARD_CSV");
        if (string.IsNullOrEmpty(path)) return;
        try
        {
            string dir = System.IO.Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir)) System.IO.Directory.CreateDirectory(dir);
            s_rewardCsv = new System.IO.StreamWriter(path, append: false) { AutoFlush = true };
            s_rewardCsv.WriteLine("agent,episode,steps,r1,r3,r4,r5,r6,r8,shaping,terminal,total,end");
            UnityEngine.Debug.Log($"[reward csv] per-episode reward terms -> {path}");
        }
        catch (System.Exception e) { UnityEngine.Debug.LogWarning($"[reward csv] init failed: {e.Message}"); }
    }

    void WriteRewardRow()
    {
        if (s_rewardCsv == null) return;
        float shaping = epR1 + epR3 + epR4 + epR5 + epR6 + epR8;
        float total   = shaping + epTerm;
        string reason = (lastEndReason ?? "").Split('(')[0].Trim().Replace(",", ";");
        lock (s_rewardCsvLock)
        {
            s_rewardCsv.WriteLine(
                $"{GetInstanceID()},{episodeCount},{epSteps}," +
                $"{epR1:F3},{epR3:F3},{epR4:F3},{epR5:F3},{epR6:F3},{epR8:F3}," +
                $"{shaping:F3},{epTerm:F3},{total:F3},{reason}");
        }
    }

    void SetWheelVelocity(ArticulationBody wheel, float radPerSec)
    {
        if (wheel == null) return;

        float capRad = maxWheelVelocityDeg * Mathf.Deg2Rad;
        radPerSec = Mathf.Clamp(radPerSec, -capRad, capRad);
        var drive = wheel.xDrive;
        drive.targetVelocity = radPerSec * Mathf.Rad2Deg;
        drive.forceLimit = wheelForceLimit;
        drive.damping = wheelDamping;
        drive.stiffness = 0f;
        wheel.xDrive = drive;
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

        EnsureReady();

        if (episodeCount > 0)
        {
            Log($"Episode {episodeCount} ended: {lastEndReason} | box-touch {boxTouchTotal:F1} over {boxTouchSteps} steps");
            EnsureRewardCsv();
            WriteRewardRow();
        }
        epR1 = epR3 = epR4 = epR5 = epR6 = epR8 = epTerm = 0f; epSteps = 0;
        boxTouchTotal = 0f; boxTouchSteps = 0;
        episodeCount++;

        lastEndReason = "TIMEOUT (max steps)";

        Log($"Episode {episodeCount} started");

        collisionHandled = false;
        firstActionLogged = false;

        for (int i = 0; i < 6; i++)
        {
            armTargets[i] = JointHomeDeg(i);
            SetJointTarget(armJoints[i], armTargets[i]);
        }

        snapStepsLeft = hardSnapSteps;

        SpawnTarget();
        PlaceSupportBox();

        prevDistance = Vector3.Distance(GripperPosition(), target.position);
        prevPhi = PoseAngleErrorRad();

        innerBandAwarded = false;
        outerBandAwarded = false;
        midBandAwarded = false;
        farBandAwarded = false;

        havePrevActions = false;
        for (int i = 0; i < prevActions.Length; i++) prevActions[i] = 0f;

    }

    void FixedUpdate()
    {
        if (snapStepsLeft > 0)
        {
            snapStepsLeft--;
            SnapArmToHome(false);
            SetWheelVelocity(frontLeftWheel, 0f);
            SetWheelVelocity(frontRightWheel, 0f);
            SetWheelVelocity(rearLeftWheel, 0f);
            SetWheelVelocity(rearRightWheel, 0f);

            if (snapStepsLeft == 0 && !baselineEstablished)
                baselinePending = true;

            return;
        }

        if (baselinePending)
        {
            EstablishCollisionBaseline();
            baselineEstablished = true;
            baselinePending = false;
            return;
        }

        if (!collisionHandled)
        {
            CheckSelfCollisions();
            if (!collisionHandled)
                CheckEnvironmentCollisions();
            if (!collisionHandled && baseBoxCollisionFatal)
                CheckBaseBoxCollision();
        }

    }

    void SnapArmToHome(bool verbose)

    {
        for (int i = 0; i < 6; i++)
        {
            ArticulationBody j = armJoints[i];
            if (j == null) continue;
            if (j.dofCount != 1)
            {
                if (verbose)
                    Debug.LogWarning($"[ArmAgentL_6dof] armJoints[{i}] '{j.name}' has dofCount={j.dofCount}, expected 1 - this joint is NOT being reset.");
                continue;
            }
            j.jointPosition = new ArticulationReducedSpace(JointHomeDeg(i) * Mathf.Deg2Rad);
            j.jointVelocity = new ArticulationReducedSpace(0f);
            j.jointForce    = new ArticulationReducedSpace(0f);
        }

    }

    void ZeroLinkVelocities(ArticulationBody root)
    {
        foreach (var b in root.GetComponentsInChildren<ArticulationBody>())
        {
            b.velocity = Vector3.zero;
            b.angularVelocity = Vector3.zero;
        }
    }

    void SpawnTarget()
    {

        if (useEvalSamples)
        {
            int nx = Mathf.Max(1, evalXSteps), ny = Mathf.Max(1, evalYSteps), nz = Mathf.Max(1, evalZSteps);
            int total = nx * ny * nz;
            int k = evalSampleIdx % total;
            int zi = k / (nx * ny), yi = (k % (nx * ny)) / nx, xi = k % nx;
            float xf = (nx <= 1) ? evalXMin : Mathf.Lerp(evalXMin, evalXMax, xi / (float)(nx - 1));
            float yf = (ny <= 1) ? evalYMin : Mathf.Lerp(evalYMin, evalYMax, yi / (float)(ny - 1));
            float zf = (nz <= 1) ? evalZMin : Mathf.Lerp(evalZMin, evalZMax, zi / (float)(nz - 1));
            target.position = startPosition + new Vector3(xf, yf, zf);
            target.rotation = Quaternion.identity;
            Debug.Log($"[s9] eval sample {k + 1}/{total}: offset=({xf:F2}, {yf:F2}, {zf:F2})");
            evalSampleIdx++;
            return;
        }

        if (useTArcCurriculum)
        {

            Vector3 tpos; Quaternion trot;
            if (tArcLineArc)
            {
                tArc.SampleLineArc(startPosition, Quaternion.identity, tArcEpisodeIdx, tArcLineArcCount, tArcLineArcLineCount, out tpos, out trot);
                tArcEpisodeIdx++;
            }
            else if (tArcFullRange)
                tArc.SampleFullRange(startPosition, Quaternion.identity, out tpos, out trot);
            else
                tArc.Sample(startPosition, Quaternion.identity, Academy.Instance.TotalStepCount, out tpos, out trot);
            target.position = tpos;
            target.rotation = trot;
            return;
        }

        Vector3 offset = targetSpawnOffset;
        if (randomizeTargetPosition)
        {
            float frac = 1f;
            if (useSpawnCurriculum)
            {

                float p = Mathf.Clamp01(Academy.Instance.TotalStepCount / (float)Mathf.Max(spawnCurriculumFullSteps, 1));
                frac = spawnCurriculumStartFrac + (1f - spawnCurriculumStartFrac) * p;
            }

            int attempts = 0;
            do
            {
                offset = targetSpawnOffset;
                offset.x += Random.Range(-1f, 1f) * frac * spawnRangeX;
                offset.z += Random.Range(-1f, 1f) * frac * spawnRangeZ;
                attempts++;
            }
            while (minSpawnDistance > 0f
                   && new Vector2(offset.x, offset.z).magnitude < minSpawnDistance
                   && attempts < maxSpawnAttempts);
            if (minSpawnDistance > 0f && new Vector2(offset.x, offset.z).magnitude < minSpawnDistance)
                Log($"spawn-rejection: could not reach minSpawnDistance={minSpawnDistance:F2} m in {maxSpawnAttempts} tries "
                    + $"(patch too tight at frac={frac:F2}); kept d={new Vector2(offset.x, offset.z).magnitude:F2} m");
        }
        target.position = startPosition + offset;

        if (randomizeTargetOrientation)
        {
            float ofrac = 1f;
            if (useOrientCurriculum)
            {
                float p = Mathf.Clamp01(Academy.Instance.TotalStepCount / (float)Mathf.Max(orientCurriculumFullSteps, 1));
                ofrac = orientCurriculumStartFrac + (1f - orientCurriculumStartFrac) * p;
            }
            float ry = Random.Range(-1f, 1f) * ofrac * orientTiltRangeDeg;
            target.rotation = Quaternion.Euler(0f, ry, 0f);
        }
        else
        {
            target.rotation = Quaternion.identity;
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (baseLink == null)
        {
            Debug.LogError("[ArmAgentL_6dof] baseLink is not assigned. Drag the chassis Transform into the Inspector slot.");
            for (int i = 0; i < 14; i++) sensor.AddObservation(0f);
            return;
        }

        Vector3 gripperLocal = WorldToLocalPoint(baseLink, GripperPosition());
        Vector3<FLU> gFlu = gripperLocal.To<FLU>();

        Quaternion gRot = LocalRotation(GripperRotation());

        Vector3 targetLocal = WorldToLocalPoint(baseLink, target.position);
        Vector3<FLU> tFlu = targetLocal.To<FLU>();

        Quaternion tRot = LocalRotation(TargetAlignedRotation());

        sensor.AddObservation(gFlu.x);
        sensor.AddObservation(gFlu.y);
        sensor.AddObservation(gFlu.z);
        sensor.AddObservation(gRot.x);
        sensor.AddObservation(gRot.y);
        sensor.AddObservation(gRot.z);
        sensor.AddObservation(gRot.w);
        sensor.AddObservation(tFlu.x);
        sensor.AddObservation(tFlu.y);
        sensor.AddObservation(tFlu.z);
        sensor.AddObservation(tRot.x);
        sensor.AddObservation(tRot.y);
        sensor.AddObservation(tRot.z);
        sensor.AddObservation(tRot.w);
    }

    Quaternion TargetAlignedRotation()
    {
        Quaternion q = (target != null) ? target.rotation : Quaternion.identity;
        return q * Quaternion.Euler(targetFrameOffsetEuler);
    }

    Quaternion RelativePoseRotation()
    {
        if (!HasFingers || target == null) return Quaternion.identity;
        Quaternion q = Quaternion.Inverse(GripperRotation()) * TargetAlignedRotation();
        if (q.w < 0f) { q.x = -q.x; q.y = -q.y; q.z = -q.z; q.w = -q.w; }
        return q;
    }

    float RotationErrorAngleRad(Quaternion rotA, Quaternion rotB)
    {

        Matrix4x4 Ra = Matrix4x4.Rotate(rotA);
        Matrix4x4 Rb = Matrix4x4.Rotate(rotB);

        Matrix4x4 Rerr = Ra.transpose * Rb;

        float trace = Rerr.m00 + Rerr.m11 + Rerr.m22;
        float cosTheta = Mathf.Clamp((trace - 1f) * 0.5f, -1f, 1f);
        return Mathf.Acos(cosTheta);
    }

    float PoseAngleErrorRad()
    {
        if (!HasFingers || target == null) return 0f;
        return RotationErrorAngleRad(GripperRotation(), TargetAlignedRotation());
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {

        bool frozen = (snapStepsLeft > 0);

        int available = actionBuffers.ContinuousActions.Length;

        if (available != NumActions && !actionMismatchLogged)
        {
            actionMismatchLogged = true;
            Log($"ACTION MISMATCH: Behavior Parameters gives {available} continuous action(s), expected {NumActions} " +
                $"(6 arm joints + base v, omega). Set 'Continuous Actions' = {NumActions} on the Behavior Parameters " +
                $"component -- the serialized Inspector value governs the build.");
        }

        if (!frozen && !firstActionLogged)
        {
            firstActionLogged = true;
            float a0     = available > 0 ? actionBuffers.ContinuousActions[0] : float.NaN;
            float aBaseV = available > 6 ? actionBuffers.ContinuousActions[6] : float.NaN;
            float aBaseW = available > 7 ? actionBuffers.ContinuousActions[7] : float.NaN;
            Log($"first action: count={available} (expected {NumActions}: 6 arm + base v,w), " +
                $"arm[0](J1)={a0:F3}, base v={aBaseV:F3}, base w={aBaseW:F3}, " +
                $"J1angle={JointDeg(0):F1}, J2angle={JointDeg(1):F1}");
        }

        float rCtrl = 0f;
        float rRate = 0f;

        if (!frozen)
        {
            float stepDeltaDeg = maxJointDeltaDeg;
            if (adaptiveActionScale)
            {
                float prox = Mathf.Clamp01(1f - Mathf.Max(prevDistance / Mathf.Max(fineDistance, 1e-6f),
                                                          prevPhi      / Mathf.Max(fineOrient,   1e-6f)));
                stepDeltaDeg = maxJointDeltaDeg * Mathf.Lerp(1f, fineDeltaFrac, prox);
            }

            float maxJvRad = maxJointVelocityDeg * Mathf.Deg2Rad;
            foreach (var jj in armJoints) if (jj != null) jj.maxJointVelocity = maxJvRad;

            for (int k = 0; k < 6; k++)
            {
                int i = controlledJointIndices[k];
                float a = actionBuffers.ContinuousActions[k];

                rCtrl += a * a;
                if (havePrevActions) { float da = a - prevActions[k]; rRate += da * da; }
                prevActions[k] = a;

                armTargets[i] = Mathf.Clamp(armTargets[i] + a * stepDeltaDeg,
                                            jointMinDeg[i], jointMaxDeg[i]);

                SetJointTarget(armJoints[i], armTargets[i]);
            }

            if (available >= NumActions)
            {
                float v_p = actionBuffers.ContinuousActions[6];
                float w_p = actionBuffers.ContinuousActions[7];

                float v_u = new Vector3<FLU>(v_p, 0f, 0f).toUnity.z;
                float w_u = new Vector3<FLU>(0f, 0f, w_p).toUnityAngularVelocity.y;

                float v = v_u * maxLinearVel;
                float w = -w_u * maxAngularVel;

                float s = 1f;
                if (Mathf.Abs(v) > maxLinearVel)  s = Mathf.Min(s, maxLinearVel  / Mathf.Abs(v));
                if (Mathf.Abs(w) > maxAngularVel) s = Mathf.Min(s, maxAngularVel / Mathf.Abs(w));
                v = Mathf.Clamp(v * s, -maxLinearVel,  maxLinearVel);
                w = Mathf.Clamp(w * s, -maxAngularVel, maxAngularVel);

                if (havePrevActions)
                {
                    float dv = v_p - prevActions[6]; float dw = w_p - prevActions[7];
                    rRate += dv * dv + dw * dw;
                }
                prevActions[6] = v_p; prevActions[7] = w_p;

                float halfL  = wheelSeparation * 0.5f;
                float vLeft  = (v - w * halfL) / wheelRadius;
                float vRight = (v + w * halfL) / wheelRadius;

                SetWheelVelocity(frontLeftWheel,  -vLeft);
                SetWheelVelocity(rearLeftWheel,   -vLeft);
                SetWheelVelocity(frontRightWheel,  vRight);
                SetWheelVelocity(rearRightWheel,   vRight);
            }

            havePrevActions = true;

            if (traceJoints && traceEveryNSteps > 0 && StepCount % traceEveryNSteps == 0)
                Log($"[trace step {StepCount}] {JointTrace(0)}  ||  {JointTrace(1)}");
        }

        Vector3 deltaT = target.position - GripperPosition();
        float d = deltaT.magnitude;

        float phi = PoseAngleErrorRad();

        bool reached = (d < successRadius) && (phi < orientToleranceRad);

        float r1 = distanceRewardScale * Mathf.Clamp(prevDistance - d, -progressClamp, progressClamp);
        float r3 = (d < successRadius) ? 0f : ((d < prevDistance) ? directionalReward : -directionalReward);

        float r4 = 0f;
        if (d <= regionFarRadius   && !farBandAwarded)   { r4 += regionFarReward;   farBandAwarded   = true; }
        if (d <= regionMidRadius   && !midBandAwarded)   { r4 += regionMidReward;   midBandAwarded   = true; }
        if (d <= regionOuterRadius && !outerBandAwarded) { r4 += regionOuterReward; outerBandAwarded = true; }
        if (d <= regionInnerRadius && !innerBandAwarded) { r4 += regionInnerReward; innerBandAwarded = true; }
        float r5 = orientRewardScale * (Mathf.Exp(-phi) - 1f);
        float r6 = (phi < orientToleranceRad) ? 0f : ((phi < prevPhi) ? orientDirectionalReward : -orientDirectionalReward);

        float r8 = -actionRatePenaltyScale * (rRate / (4f * 8f));

        AddReward(r1 + r3 + r4 + r5 + r6 + r8);
        epR1 += r1; epR3 += r3; epR4 += r4; epR5 += r5; epR6 += r6; epR8 += r8; epSteps++;

        prevDistance = d;
        prevPhi = phi;

        if (reached)
        {
            AddReward(reachReward);
            epTerm = reachReward;
            lastEndReason = $"SUCCESS (d={d:F3}, phi={phi:F3}rad, r1={r1:F3} r5={r5:F3}, reach=+{reachReward:F0}, J1={JointDeg(0):F1})";
            EndEpisode();
            return;
        }

        Transform chassis = baseLink != null ? baseLink : (baseBody != null ? baseBody.transform : transform);
        float chassisY = chassis.position.y;
        float tilt = Vector3.Angle(chassis.up, Vector3.up);
        if (chassisY < fallY || tilt > maxTiltDeg)
        {
            lastEndReason = $"FAIL base fell (y={chassisY:F2}, tilt={tilt:F0}deg)";
            AddReward(-fallPenalty);
            epTerm = -fallPenalty;
            EndEpisode();
            return;
        }

        if (MaxStep > 0 && StepCount >= MaxStep)
        {
            lastEndReason = $"TIMEOUT (d={d:F3}, J1={JointDeg(0):F1}/tgt{armTargets[0]:F0}, J2={JointDeg(1):F1}/tgt{armTargets[1]:F0})";
            EndEpisode();
        }
    }

    float JointDeg(int i)
    {
        var j = armJoints[i];
        return (j != null && j.dofCount == 1) ? j.jointPosition[0] * Mathf.Rad2Deg : float.NaN;
    }

    string JointTrace(int i)
    {
        var j = armJoints[i];
        if (j == null || j.dofCount != 1) return $"J{i + 1}=<no 1-dof joint>";
        float angle = j.jointPosition[0] * Mathf.Rad2Deg;
        float vel   = j.jointVelocity[0] * Mathf.Rad2Deg;
        float force = j.jointForce[0];
        return $"J{i + 1} tgt={armTargets[i]:F1} xdrv={j.xDrive.target:F1} angle={angle:F1} " +
            $"vel={vel:F1}/s force={force:F0}/{j.xDrive.forceLimit:F0}";
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;
        for (int i = 0; i < ca.Length; i++) ca[i] = 0f;
    }

    void SetJointTarget(ArticulationBody joint, float targetDeg)
    {
        if (joint == null) return;
        var drive = joint.xDrive;
        drive.stiffness  = armStiffness;
        drive.damping    = armDamping;
        drive.forceLimit = armForceLimit;
        drive.target     = targetDeg;
        joint.xDrive     = drive;
    }

    void OnDrawGizmos()
    {
        if (target == null || !HasFingers)
            return;

        Gizmos.color = Color.red;
        Gizmos.DrawLine(target.position, GripperPosition());
        if (drawFrameGizmos)
        {
            DrawFrameGizmo(GripperPosition(), GripperRotation(), gizmoAxisLength);
            DrawFrameGizmo(target.position, target.rotation, gizmoAxisLength);
        }

    #if UNITY_EDITOR

        Vector3 mid = (target.position + GripperPosition()) * 0.5f;
        float distance = Vector3.Distance(target.position, GripperPosition());
        UnityEditor.Handles.Label(mid, distance.ToString("F2") + " m");
    #endif
    }

    void DrawFrameGizmo(Vector3 pos, Quaternion rot, float len)
    {
    #if UNITY_EDITOR
        float t = gizmoAxisThickness;

        UnityEditor.Handles.color = Color.red;
        UnityEditor.Handles.DrawLine(pos, pos + rot * Vector3.right   * len, t);

        UnityEditor.Handles.color = Color.green;
        UnityEditor.Handles.DrawLine(pos, pos + rot * Vector3.up      * len, t);

        UnityEditor.Handles.color = Color.blue;
        UnityEditor.Handles.DrawLine(pos, pos + rot * Vector3.forward * len, t);

    #else
        Gizmos.color = Color.red;   Gizmos.DrawLine(pos, pos + rot * Vector3.right   * len);
        Gizmos.color = Color.green; Gizmos.DrawLine(pos, pos + rot * Vector3.up      * len);
        Gizmos.color = Color.blue;  Gizmos.DrawLine(pos, pos + rot * Vector3.forward * len);
    #endif
    }
}
