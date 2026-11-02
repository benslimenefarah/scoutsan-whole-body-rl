using UnityEngine;

public class GripperFrameProbe : MonoBehaviour
{
    [Header("Gripper fingers (drag the SAME two as on the ScoutsanV1_1b agent)")]
    public Transform fingerLink1;
    public Transform fingerLink2;

    [Header("Frame offsets (copy the SAME values from the ScoutsanV1_1b agent)")]
    [Tooltip("Rotation offset applied to fingerLink1.rotation to get the gripper frame (agent: gripperFrameEuler).")]
    public Vector3 gripperFrameEuler = Vector3.zero;
    [Tooltip("Tip offset expressed IN the gripper frame (agent default (0, 0.02, 0)).")]
    public Vector3 gripperTipOffset = new Vector3(0f, 0.02f, 0f);

    Quaternion GripperRotation()
    {
        Quaternion baseRot = fingerLink1 != null ? fingerLink1.rotation
                           : (fingerLink2 != null ? fingerLink2.rotation : transform.rotation);
        return baseRot * Quaternion.Euler(gripperFrameEuler);
    }

    Vector3 GripperPosition()
    {
        Vector3 mid = (fingerLink1 != null && fingerLink2 != null)
                      ? 0.5f * (fingerLink1.position + fingerLink2.position)
                      : (fingerLink1 != null ? fingerLink1.position : transform.position);
        return mid + GripperRotation() * gripperTipOffset;
    }

    void LateUpdate()
    {
        if (fingerLink1 == null && fingerLink2 == null) return;
        transform.SetPositionAndRotation(GripperPosition(), GripperRotation());
    }
}
