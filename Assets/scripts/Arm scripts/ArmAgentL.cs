using System.IO;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;

public class ArmAgentL : Agent
{

    public ArticulationBody[] armJoints = new ArticulationBody[6];

    public Transform fingerLink1;
    public Transform fingerLink2;

    public Vector3 gripperFrameEuler = Vector3.zero;

    public Vector3 gripperTipOffset = new Vector3(0f, 0.02f, 0f);

    public Transform target;

    public Transform baseLink;

    public int hardSnapSteps = 3;

    public int[] controlledJointIndices = { 0, 1, 2, 3, 4, 5 };

    int controlledJointCount => controlledJointIndices != null ? controlledJointIndices.Length : 0;

    public bool holdLShape = true;

    public bool traceJoints = true;
    public int traceEveryNSteps = 20;

    public float maxJointDeltaDeg = 3f;

    public bool  adaptiveActionScale = true;
    public float fineDistance  = 0.20f;
    public float fineOrient    = 0.30f;
    public float fineDeltaFrac = 0.03f;

    public float[] jointMinDeg  = { -360f, -95f, -135f, -360f, -135f, -360f };
    public float[] jointMaxDeg  = {  360f,  95f,  135f,  360f,  135f,  360f };

    public float[] jointHomeDeg = {    0f,   0f,     0f,    0f,    0f,    0f };

    public float armStiffness  = 10000f;
    public float armDamping    = 200f;
    public float armForceLimit = 1000f;
    public float maxJointVelocityDeg = 40.1f;

    [Header("Spawn shell radii (target sampled between these -- tune in the Inspector)")]
    public Transform sphereCenter;
    [Tooltip("Inner shell radius (m)")]
    public float radius1 = 0.042f;
    [Tooltip("Outer shell radius (m)")]
    public float radius2 = 0.776f;

    public Transform vFrame;
    public Vector3 vCenter = Vector3.zero;
    public float vHalfAngleDeg = 60f;
    public float vLength = 1.6f;
    public float vDown = 1.5f;
    public float vUp = 1.5f;

    [Header("Spawn exclusion box (targets never spawn inside)")]

    public Transform boxFrame;
    public Vector3 boxCenter = Vector3.zero;
    public Vector3 boxSize = new Vector3(0.6f, 0.4f, 0.6f);

    [Header("Eval targets (deterministic N-target sweep -- for videos / eval)")]
    public bool useEvalTargets = false;
    public int  evalTargetCount = 10;
    public int  evalTargetSeed  = 12345;
    List<Vector3>    evalPos;
    List<Quaternion> evalRot;
    int              evalIndex;

    [Header("Reach-sphere spawn (ported from ArmAgent_vf)")]
    public Transform armBase;
    public float armReachRadius  = 0.9f;
    public float minReachRadius  = 0.2f;
    public float minTargetHeight = 0.45f;
    public float maxTargetHeight = 5f;
    public float minTargetX = 0f;
    public float minTargetZ = 0.45f;

    [Header("Spawn preview (Editor gizmo -- SEE the target area to tune it, before training/watching)")]
    [Tooltip("Draw a cloud of sampled spawn points (SampleReachSpherePoint) + the reach sphere around armBase. Select the agent in the scene to see it. Adjust the fields above and the cloud updates live. Purely editor-visual; no effect on training.")]
    public bool drawSpawnSamples = false;
    [Range(50, 3000)] public int spawnSampleCount = 500;
    [Tooltip("Fixed RNG seed so the preview cloud is STABLE (only changes when you change the spawn params, not every repaint).")]
    public int spawnSampleSeed = 12345;

    [Header("Success thresholds (tune in the Inspector)")]
    [Tooltip("Position success radius (m): gripper must be within this of the target")]
    public float successRadius = 0.03f;

    public float distanceRewardScale = 2.0f;
    public float reachReward         = 30.0f;
    public float directionalReward   = 0.003f;
    public float orientDirectionalReward = 0.01f;
    [Tooltip("Inner reward-band radius (m)")]
    public float regionInnerRadius   = 0.05f;
    public float regionInnerReward   = 5.0f;
    [Tooltip("Outer reward-band radius (m)")]
    public float regionOuterRadius   = 0.10f;
    public float regionOuterReward   = 1.0f;
    public float orientRewardScale  = 4.0f;
    [Tooltip("Orientation success tolerance (rad)")]
    public float orientToleranceRad = 0.05f;

    [Header("Orientation milestone band (confondu)")]
    [Tooltip("Inner orientation band (rad): crossing into it pays orientInnerReward once per episode")]
    public float orientInnerToleranceRad = 0.08f;
    public float orientInnerReward       = 8.0f;
    [Tooltip("Outer orientation band (rad): crossing into it pays orientOuterReward once per episode")]
    public float orientOuterToleranceRad = 0.25f;
    public float orientOuterReward       = 3.0f;

    [Header("Trajectory smoothing (control-action penalty)")]
    [Tooltip("w_ctrl: weight on sum of squared actions, subtracted from the reward each step (0 = off)")]
    public float controlActionRewardScale = 0.01f;

    [Tooltip("w_rate: weight on sum of squared action DIFFERENCES (a_t - a_{t-1})^2, subtracted each step (0 = off)")]
    public float actionRatePenaltyScale = 0.02f;

    public Vector3 targetFrameOffsetEuler = Vector3.zero;

    [Tooltip("Per-episode target yaw (deg) about the vertical axis (world Y = your z-up). alpha ~ U(-range,+range); z stays up, only the heading varies. Narrow start = 20 deg. SERIALIZED in the scene -> Inspector value governs the build.")]
    public float targetYawRangeDeg = 20f;
    Quaternion targetBaseRotation = Quaternion.identity;

    public bool drawFrameGizmos = true;

    public float gizmoAxisLength = 0.3f;

    public float gizmoAxisThickness = 6f;

    public float floorY = 0f;

    public float collisionPenalty = 50.0f;

    public string[] collisionExclusions = { "scoutsan_box_1" };

    public string[] gripperSelfCollisionGroup = { "base_link_0", "finger1_Link_0", "finger2_Link_0" };

    ArticulationBody armRoot;
    Vector3 armRootStartPos;
    Quaternion armRootStartRot;

    float[] armTargets = new float[6];

    readonly float[] prevActions = new float[6];
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
    bool innerOrientAwarded = false;
    bool outerOrientAwarded = false;

    readonly List<Collider> armColliders = new List<Collider>();
    readonly HashSet<Collider> armColliderSet = new HashSet<Collider>();
    readonly HashSet<(int, int)> baselineSelfPairs = new HashSet<(int, int)>();
    readonly HashSet<(int, int)> baselineEnvPairs = new HashSet<(int, int)>();
    readonly Collider[] overlapBuffer = new Collider[32];

    StreamWriter logWriter;
    int episodeCount = 0;
    string lastEndReason = "n/a";

    public override void Initialize()
    {
        Time.timeScale = 1f;

        string logDir = Path.Combine(Application.dataPath, "..", "Logs");
        Directory.CreateDirectory(logDir);

        string logPath = Path.Combine(logDir, $"arm_agent_l_{System.DateTime.Now:yyyyMMdd_HHmmss}_{GetInstanceID()}.log");
        logWriter = new StreamWriter(logPath, append: false) { AutoFlush = true };
        Log($"=== training log started: {logPath} ===");

        targetBaseRotation = (target != null) ? target.rotation : Quaternion.identity;
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
        Debug.Log($"[ArmAgentL_6dof] {message}");
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

        armRoot = RootOf(armJoints[0]);

        if (armRoot == null) { Debug.LogError("ArmAgentL: could not find the arm's articulation root."); return; }

        armRoot.useGravity = false;

        armRootStartPos = armRoot.transform.position;
        armRootStartRot = armRoot.transform.rotation;

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
                    FailEpisode($"'{col.name}' hit environment '{hit.name}'");
                    return;
                }
            }
        }
    }

    void FailEpisode(string reason)

    {
        if (collisionHandled) return;
        collisionHandled = true;
        lastEndReason = $"FAIL: {reason}";
        Log($"Collision failure: {reason} -- penalty {-collisionPenalty}, ending episode.");
        AddReward(-collisionPenalty);
        EndEpisode();
    }

    public override void OnEpisodeBegin()

    {
        EnsureReady();

        if (episodeCount > 0)
            Log($"Episode {episodeCount} ended: {lastEndReason}");
        episodeCount++;

        lastEndReason = "TIMEOUT (max steps)";

        Log($"Episode {episodeCount} started");

        collisionHandled = false;
        firstActionLogged = false;

        if (armRoot != null)
        {
            armRoot.TeleportRoot(armRootStartPos, armRootStartRot);
            armRoot.velocity = Vector3.zero;
            armRoot.angularVelocity = Vector3.zero;

            armRoot.immovable = false;

        }

        for (int i = 0; i < 6; i++)
        {
            armTargets[i] = jointHomeDeg[i];
            SetJointTarget(armJoints[i], armTargets[i]);
        }

        SnapArmToHome(true);
        snapStepsLeft = hardSnapSteps;

        SpawnTarget();

        prevDistance = Vector3.Distance(GripperPosition(), target.position);
        prevPhi = PoseAngleErrorRad();

        innerBandAwarded = false;
        outerBandAwarded = false;

        havePrevActions = false;
        for (int i = 0; i < prevActions.Length; i++) prevActions[i] = 0f;
    }

    void FixedUpdate()
    {
        if (snapStepsLeft > 0)
        {
            snapStepsLeft--;
            SnapArmToHome(false);

            if (snapStepsLeft == 0 && armRoot != null)
                armRoot.immovable = true;

            if (snapStepsLeft == 0 && !baselineEstablished)
                baselinePending = true;

            return;
        }

        if (holdLShape) HoldLShape();

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
        }
    }

    void HoldLShape()
    {
        for (int i = 0; i < 6; i++)
        {
            if (System.Array.IndexOf(controlledJointIndices, i) >= 0) continue;
            ArticulationBody j = armJoints[i];
            if (j == null || j.dofCount != 1) continue;
            j.jointPosition = new ArticulationReducedSpace(jointHomeDeg[i] * Mathf.Deg2Rad);
            j.jointVelocity = new ArticulationReducedSpace(0f);
            j.jointForce    = new ArticulationReducedSpace(0f);
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
            j.jointPosition = new ArticulationReducedSpace(jointHomeDeg[i] * Mathf.Deg2Rad);
            j.jointVelocity = new ArticulationReducedSpace(0f);
            j.jointForce    = new ArticulationReducedSpace(0f);
        }

        if (armRoot != null) ZeroLinkVelocities(armRoot);
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

        if (useEvalTargets)
        {
            int n = Mathf.Max(1, evalTargetCount);
            if (evalPos == null || evalPos.Count != n) BuildEvalTargets();
            int i = ((evalIndex % n) + n) % n;
            target.position = evalPos[i];
            target.rotation = evalRot[i];
            evalIndex++;
            return;
        }

        target.position = SampleOnePoint();

        float alphaDeg = Random.Range(-targetYawRangeDeg, targetYawRangeDeg);
        target.rotation = Quaternion.AngleAxis(alphaDeg, Vector3.up) * targetBaseRotation;
    }

    Vector3 SampleOnePoint()
    {
        return SampleShellPoint();
    }

    void BuildEvalTargets()
    {
        int n = Mathf.Max(1, evalTargetCount);
        evalPos = new List<Vector3>(n);
        evalRot = new List<Quaternion>(n);
        var prev = Random.state;
        Random.InitState(evalTargetSeed);
        for (int k = 0; k < n; k++)
        {
            Vector3 pos = SampleShellPoint();
            float alphaDeg = Random.Range(-targetYawRangeDeg, targetYawRangeDeg);
            evalPos.Add(pos);
            evalRot.Add(Quaternion.AngleAxis(alphaDeg, Vector3.up) * targetBaseRotation);
        }
        Random.state = prev;
        evalIndex = 0;
        Debug.Log($"[ArmAgentL] eval targets built: {n} fixed targets (seed {evalTargetSeed}).");
    }

    Vector3 SampleReachSpherePoint()
    {
        Transform ab = armBase != null ? armBase : (baseLink != null ? baseLink : transform);
        Vector3 center = ab.position;
        Vector3 candidate;
        int safety = 0;
        do
        {
            float y = Random.Range(minTargetHeight, maxTargetHeight);
            float dy = y - center.y;
            float maxHorizontal = Mathf.Sqrt(Mathf.Max(0f, armReachRadius * armReachRadius - dy * dy));
            float minHorizontal = Mathf.Min(minReachRadius, maxHorizontal);
            float horizontalRadius = Random.Range(minHorizontal, maxHorizontal);
            float angle = Random.Range(-Mathf.PI * 0.5f, Mathf.PI * 0.5f);
            float xMag = Mathf.Max(Mathf.Abs(Mathf.Cos(angle) * horizontalRadius), minTargetX);
            float zMag = Mathf.Max(Mathf.Abs(Mathf.Sin(angle) * horizontalRadius), minTargetZ);
            float xSign = (Random.value < 0.5f) ? -1f : 1f;
            candidate = new Vector3(center.x + xSign * xMag, y, center.z + zMag);
            safety++;
        }
        while (Vector3.Distance(candidate, center) > armReachRadius && safety < 50);
        return candidate;
    }

    Vector3 SampleShellPoint()
    {
        SpawnFrame2D(out Vector2 v, out Vector2 fwd, out Vector2 right);
        Vector3 sc = SphereWorldCenter();
        float apexY = VWorldCenter().y;
        float yMin = apexY - vDown, yMax = apexY + vUp;
        float tanHalf = Mathf.Tan(vHalfAngleDeg * Mathf.Deg2Rad);
        float rIn = Mathf.Min(radius1, radius2), rOut = Mathf.Max(radius1, radius2);
        float rIn3 = rIn * rIn * rIn, rOut3 = rOut * rOut * rOut;
        for (int i = 0; i < 64; i++)
        {

            float r = Mathf.Pow(Mathf.Lerp(rIn3, rOut3, Random.value), 1f / 3f);
            Vector3 p = sc + Random.onUnitSphere * r;
            if (p.y < yMin || p.y > yMax) continue;
            Vector2 rel = new Vector2(p.x - v.x, p.z - v.y);
            float fwdDist = Vector2.Dot(rel, fwd);
            if (fwdDist < 0f || fwdDist > vLength) continue;
            if (Mathf.Abs(Vector2.Dot(rel, right)) > fwdDist * tanHalf) continue;
            if (InExclusionBox(p)) continue;
            return p;
        }
        return sc + new Vector3(fwd.x, 0f, fwd.y) * rOut;
    }

    void SpawnFrame2D(out Vector2 v, out Vector2 fwd, out Vector2 right)
    {
        Vector3 vp = VWorldCenter();
        v = new Vector2(vp.x, vp.z);

        Transform tv = vFrame != null ? vFrame : (baseLink != null ? baseLink : transform);
        Vector3 f = tv.forward;
        Vector3 r = tv.right;
        fwd   = new Vector2(f.x, f.z);
        right = new Vector2(r.x, r.z);
        if (fwd.sqrMagnitude   < 1e-6f) fwd   = Vector2.up;
        if (right.sqrMagnitude < 1e-6f) right = Vector2.right;
        fwd.Normalize();
        right.Normalize();
    }

    Vector3 SphereWorldCenter()
    {
        return sphereCenter != null ? sphereCenter.position
             : (baseLink != null ? baseLink.position : transform.position);
    }

    Vector3 VWorldCenter()
    {
        Transform t = vFrame != null ? vFrame : (baseLink != null ? baseLink : transform);
        return t.TransformPoint(vCenter);
    }

    bool InExclusionBox(Vector3 p)
    {
        if (boxSize.x <= 0f || boxSize.y <= 0f || boxSize.z <= 0f) return false;
        Transform t = boxFrame != null ? boxFrame : (baseLink != null ? baseLink : transform);
        Vector3 local = Quaternion.Inverse(t.rotation) * (p - t.TransformPoint(boxCenter));
        Vector3 h = boxSize * 0.5f;
        return Mathf.Abs(local.x) <= h.x && Mathf.Abs(local.y) <= h.y && Mathf.Abs(local.z) <= h.z;
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

        if (available < controlledJointCount && !actionMismatchLogged)
        {

            actionMismatchLogged = true;
            Log($"ACTION MISMATCH: Behavior Parameters gives {available} continuous action(s) but " +
                $"controlledJointCount={controlledJointCount}. Set 'Continuous Actions' = {controlledJointCount} on the " +
                $"Behavior Parameters component, or only J1..J{available} will move.");
        }
        else if (available > controlledJointCount && !actionMismatchLogged)
        {

            actionMismatchLogged = true;
            Log($"ACTION MISMATCH: {available} continuous actions arrive but controlledJointIndices has {controlledJointCount} entr(ies) " +
                $"({string.Join(",", controlledJointIndices)}), so the extra actions drive no joint. " +
                $"Set 'Continuous Actions' = {controlledJointCount} on Behavior Parameters, or extend controlledJointIndices " +
                $"on the ArmAgentL_6dof component in the Inspector (the serialized Inspector value overrides the script's default).");
        }

        if (!frozen && !firstActionLogged)
        {
            firstActionLogged = true;
            float a0 = available > 0 ? actionBuffers.ContinuousActions[0] : float.NaN;
            float a1 = available > 1 ? actionBuffers.ContinuousActions[1] : float.NaN;
            int j2Dof = armJoints[1] != null ? armJoints[1].dofCount : -1;
            Log($"first action: count={available}, controlledJointIndices=[{string.Join(",", controlledJointIndices)}], " +
                $"driven joints n={Mathf.Min(controlledJointCount, available)}, " +
                $"action[0](J1)={a0:F3}, action[1](J2)={a1:F3}, " +
                $"J2 '{(armJoints[1] != null ? armJoints[1].name : "null")}' dofCount={j2Dof}, " +
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

            int n = Mathf.Min(controlledJointCount, available);
            for (int k = 0; k < n; k++)
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

            havePrevActions = true;

            if (traceJoints && traceEveryNSteps > 0 && StepCount % traceEveryNSteps == 0)
                Log($"[trace step {StepCount}] {JointTrace(0)}  ||  {JointTrace(1)}");
        }

        Vector3 deltaT = target.position - GripperPosition();
        float d = deltaT.magnitude;

        float phi = PoseAngleErrorRad();

        bool reached = (d < successRadius) && (phi < orientToleranceRad);

        float r1 = distanceRewardScale * (Mathf.Exp(-d) - 1f);

        float r2 = reached ? reachReward : 0f;

        float r3 = (d < successRadius) ? 0f
                                       : ((d < prevDistance) ? directionalReward : -directionalReward);

        float r4 = 0f;
        if (d <= regionInnerRadius)
        {
            if (!innerBandAwarded) { r4 = regionInnerReward; innerBandAwarded = true; }
        }
        else if (d <= regionOuterRadius)
        {
            if (!outerBandAwarded) { r4 = regionOuterReward; outerBandAwarded = true; }
        }

        float r5 = orientRewardScale * (Mathf.Exp(-phi) - 1f);

        float r6 = (phi < orientToleranceRad) ? 0f
                                              : ((phi < prevPhi) ? orientDirectionalReward : -orientDirectionalReward);

        float r7 = -controlActionRewardScale * rCtrl;

        float r8 = -actionRatePenaltyScale * rRate;

        AddReward(r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8);

        prevDistance = d;
        prevPhi = phi;

        if (reached)
        {

            lastEndReason = $"SUCCESS (d={d:F3}, phi={phi:F3}rad, r1={r1:F3} r4={r4:F1} r5={r5:F3} r7={r7:F3}, J1={JointDeg(0):F1}/tgt{armTargets[0]:F0})";
            EndEpisode();
        }
        else if (MaxStep > 0 && StepCount >= MaxStep)
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
        if (drawSpawnSamples) DrawSpawnSamples();

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

    void DrawSpawnSamples()
    {
        SpawnFrame2D(out Vector2 v, out Vector2 fwd, out Vector2 right);
        Vector3 sc = SphereWorldCenter();
        float apexY = VWorldCenter().y;
        float yMin = apexY - vDown, yMax = apexY + vUp;
        float tanHalf = Mathf.Tan(vHalfAngleDeg * Mathf.Deg2Rad);

        Gizmos.color = Color.green; Gizmos.DrawWireSphere(sc, radius1);
        Gizmos.color = Color.cyan;  Gizmos.DrawWireSphere(sc, radius2);

        Vector2 leftBase  = v + (fwd - right * tanHalf) * vLength;
        Vector2 rightBase = v + (fwd + right * tanHalf) * vLength;
        Gizmos.color = Color.red;
        DrawVerticalQuad(v, leftBase, yMin, yMax);
        DrawVerticalQuad(v, rightBase, yMin, yMax);
        DrawVerticalQuad(leftBase, rightBase, yMin, yMax);
        Gizmos.DrawSphere(new Vector3(v.x, 0.5f * (yMin + yMax), v.y), 0.045f);

        if (boxSize.x > 0f && boxSize.y > 0f && boxSize.z > 0f)
        {
            Transform bt = boxFrame != null ? boxFrame : (baseLink != null ? baseLink : transform);
            Matrix4x4 prevM = Gizmos.matrix;
            Gizmos.matrix = Matrix4x4.TRS(bt.TransformPoint(boxCenter), bt.rotation, Vector3.one);
            Gizmos.color = new Color(1f, 0.45f, 0.1f);
            Gizmos.DrawWireCube(Vector3.zero, boxSize);
            Gizmos.matrix = prevM;
        }

        var prev = Random.state;
        Random.InitState(spawnSampleSeed);
        Gizmos.color = Color.yellow;
        int n = Mathf.Max(1, spawnSampleCount);
        for (int i = 0; i < n; i++)
            Gizmos.DrawSphere(SampleShellPoint(), 0.02f);
        Random.state = prev;
    }

    void DrawVerticalQuad(Vector2 a, Vector2 b, float yMin, float yMax)
    {
        Vector3 a0 = new Vector3(a.x, yMin, a.y), a1 = new Vector3(a.x, yMax, a.y);
        Vector3 b0 = new Vector3(b.x, yMin, b.y), b1 = new Vector3(b.x, yMax, b.y);
        Gizmos.DrawLine(a0, b0); Gizmos.DrawLine(a1, b1);
        Gizmos.DrawLine(a0, a1); Gizmos.DrawLine(b0, b1);
    }
}
