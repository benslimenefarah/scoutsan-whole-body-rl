using UnityEngine;

public class FrameAxesRenderer : MonoBehaviour
{
    [Header("Axes (repère RGB)")]
    public float AxisLength = 0.15f;
    public float AxisWidth = 0.006f;
    [Tooltip("UNE seule couleur pour les 3 axes (ex: NOIR pour distinguer Mdes des repères RGB gripper/cible). OFF = RGB (X rouge, Y vert, Z bleu).")]
    public bool SingleColor = false;
    public Color AxisColor = Color.black;

    [Header("Sphère (optionnelle, ex: sur la cible)")]
    public bool IncludeSphere = false;
    public float SphereRadius = 0.03f;
    public Color SphereColor = new Color(1f, 1f, 0f, 1f);

    [Header("Suivi (optionnel)")]
    [Tooltip("Si défini, se cale sur la pose (position+rotation) de ce Transform -- SANS hériter son ECHELLE. À utiliser pour la CIBLE : attachez ce composant à un objet vide (échelle 1) et glissez la cible ici (la cible est un prefab mis à l'échelle 0.001 -> l'attacher directement rend les axes minuscules).")]
    public Transform FollowTransform;

    private LineRenderer[] m_axes;
    private Transform m_sphere;

    private void Start()
    {
        Shader unlit = Shader.Find("Unlit/Color");
        Color[] cols = { Color.red, Color.green, Color.blue };
        Vector3[] dirs = { Vector3.right, Vector3.up, Vector3.forward };
        string names = "XYZ";
        m_axes = new LineRenderer[3];
        for (int i = 0; i < 3; i++)
        {
            var go = new GameObject("frameAxis_" + names[i]);
            go.transform.SetParent(transform, false);
            var lr = go.AddComponent<LineRenderer>();
            lr.useWorldSpace = false;
            lr.positionCount = 2;
            lr.SetPosition(0, Vector3.zero);
            lr.SetPosition(1, dirs[i] * AxisLength);
            lr.startWidth = lr.endWidth = AxisWidth;
            lr.numCapVertices = 2;
            lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            lr.receiveShadows = false;
            Color c = SingleColor ? AxisColor : cols[i];
            lr.startColor = lr.endColor = c;
            if (unlit != null) { var m = new Material(unlit); m.color = c; lr.material = m; }
            m_axes[i] = lr;
        }

        if (IncludeSphere)
        {
            var s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            var col = s.GetComponent<Collider>();
            if (col != null) Destroy(col);
            s.name = "frameSphere";
            s.transform.SetParent(transform, false);
            s.transform.localScale = Vector3.one * (SphereRadius * 2f);
            var mr = s.GetComponent<MeshRenderer>();
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            mr.receiveShadows = false;
            if (unlit != null) { var m = new Material(unlit); m.color = SphereColor; mr.material = m; }
            m_sphere = s.transform;
        }
    }

    private void LateUpdate()
    {

        if (FollowTransform != null)
            transform.SetPositionAndRotation(FollowTransform.position, FollowTransform.rotation);
    }
}
