using UnityEngine;

[RequireComponent(typeof(Camera))]
public class TargetFollowCamera : MonoBehaviour
{
    [Header("Cibles à suivre")]
    [Tooltip("La cible à suivre (le Transform `target` de l'agent filmé).")]
    public Transform Target;
    [Tooltip("Optionnel : un Transform à la pose du gripper -> vise le MILIEU cible<->gripper. Pointez-le sur l'objet 'GripperFrame' (GripperFrameProbe).")]
    public Transform GripperPoint;

    [Header("Cadrage")]
    [Tooltip("Position de la caméra RELATIVE au point visé, en axes MONDE. Petit = gros plan. Ex (0.4,0.35,-0.6): un peu de côté, au-dessus, en retrait.")]
    public Vector3 Offset = new Vector3(0.4f, 0.35f, -0.6f);
    [Tooltip("Viser le milieu cible<->gripper (ON) ou seulement la cible (OFF).")]
    public bool LookAtMidpoint = true;
    [Tooltip("Lissage du déplacement (0 = collé à la cible sans lissage ; plus grand = plus réactif).")]
    public float FollowSmooth = 12f;

    private void LateUpdate()
    {
        if (Target == null) return;

        Vector3 look = Target.position;
        if (LookAtMidpoint && GripperPoint != null)
            look = 0.5f * (Target.position + GripperPoint.position);

        Vector3 desired = look + Offset;
        if (FollowSmooth > 0f)
        {
            float k = 1f - Mathf.Exp(-FollowSmooth * Time.deltaTime);
            transform.position = Vector3.Lerp(transform.position, desired, k);
        }
        else
        {
            transform.position = desired;
        }
        transform.LookAt(look, Vector3.up);
    }
}
