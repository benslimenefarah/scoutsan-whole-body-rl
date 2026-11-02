using UnityEngine;

[RequireComponent(typeof(Camera))]
public class EpisodeFramingCamera : MonoBehaviour
{
    [Header("Scene refs (drag from Hierarchy)")]
    public Transform target;
    public Transform baseRobot;

    [Header("Framing")]
    [Tooltip("Padding around the base<->target span (1 = exact fit, 1.4 = 40% headroom).")]
    public float margin = 1.4f;
    [Tooltip("View elevation above the ground plane, degrees.")]
    public float elevationDeg = 32f;
    [Tooltip("Yaw of the side view around the vertical axis (0 = pure side, +90 = along the axis).")]
    public float azimuthDeg = 15f;
    [Tooltip("Which side of the base->target line to view from (+1 or -1).")]
    public float sideSign = 1f;
    [Tooltip("Never get closer than this (small-separation guard), metres.")]
    public float minDistance = 4f;
    [Tooltip("Aim a little above the midpoint so the robot isn't at the frame's very bottom.")]
    public float lookHeight = 0.3f;
    [Tooltip("Target displacement (m) that counts as a NEW episode -> reframe.")]
    public float episodeMoveEpsilon = 0.1f;

    Camera m_cam;
    Vector3 m_baseStart;
    Vector3 m_lastTarget;
    bool m_haveBase;

    void Awake() { m_cam = GetComponent<Camera>(); }

    void Start()
    {
        if (baseRobot != null) { m_baseStart = baseRobot.position; m_haveBase = true; }
        if (target != null)
        {
            m_lastTarget = target.position;
            Reframe();
        }
    }

    void Update()
    {
        if (target == null) return;
        if (!m_haveBase && baseRobot != null) { m_baseStart = baseRobot.position; m_haveBase = true; }

        if ((target.position - m_lastTarget).sqrMagnitude > episodeMoveEpsilon * episodeMoveEpsilon)
        {
            m_lastTarget = target.position;
            Reframe();
        }
    }

    void Reframe()
    {
        Vector3 a = m_haveBase ? m_baseStart : transform.position;
        Vector3 b = target.position;

        float groundY = Mathf.Min(a.y, b.y);
        a.y = groundY; b.y = groundY;
        Vector3 mid = 0.5f * (a + b);

        Vector3 ab = b - a; ab.y = 0f;
        float sep = ab.magnitude;
        Vector3 abN = sep > 1e-3f ? ab / sep : Vector3.forward;

        Vector3 side = Vector3.Cross(Vector3.up, abN) * Mathf.Sign(sideSign == 0f ? 1f : sideSign);
        Vector3 horizDir = Quaternion.AngleAxis(azimuthDeg, Vector3.up) * side;

        float vHalf = Mathf.Max(1f, m_cam.fieldOfView) * 0.5f * Mathf.Deg2Rad;
        float hHalf = Mathf.Atan(Mathf.Tan(vHalf) * Mathf.Max(0.1f, m_cam.aspect));
        float needed = (sep * 0.5f * margin) / Mathf.Max(0.01f, Mathf.Tan(hHalf));
        float dist = Mathf.Max(minDistance, needed);

        float elevRad = elevationDeg * Mathf.Deg2Rad;
        Vector3 viewDir = (horizDir * Mathf.Cos(elevRad) + Vector3.up * Mathf.Sin(elevRad)).normalized;

        transform.position = mid + viewDir * dist;
        transform.LookAt(mid + Vector3.up * lookHeight);
    }
}
