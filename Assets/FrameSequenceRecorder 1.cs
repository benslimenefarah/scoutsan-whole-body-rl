using System.IO;
using UnityEngine;

[RequireComponent(typeof(Camera))]
public class FrameSequenceRecorder : MonoBehaviour
{
    [Header("Sortie")]
    public string OutputDirectory = "Recordings/frame_sequence";
    [Tooltip("Ajoute /<nom du GameObject caméra> au dossier -> deux caméras n'écrasent pas leurs images.")]
    public bool AppendCameraName = true;
    public enum ImageFormat { JPG, PNG }
    public ImageFormat Format = ImageFormat.JPG;
    [Range(1, 100)] public int JpgQuality = 90;

    [Header("Cadence et résolution")]
    public int Fps = 30;
    public int Width = 1920;
    public int Height = 1080;

    [Header("Durée")]
    [Tooltip("0 = illimité : arrêter manuellement avec StopRecording() (ou en sortant du Play)")]
    public float Duration = 0f;

    [Header("Démarrage")]
    public bool RecordOnStart = true;

    public bool IsRecording { get; private set; }
    public int FramesWritten { get; private set; }

    private Camera m_cam;
    private RenderTexture m_rt;
    private Texture2D m_tex;
    private int m_frame;
    private string m_dir;

    private static int s_active = 0;
    private static int s_savedCaptureFramerate = 0;

    private void Awake()
    {
        m_cam = GetComponent<Camera>();
    }

    private void Start()
    {
        if (RecordOnStart)
            BeginRecording();
    }

    public void BeginRecording()
    {
        if (IsRecording) return;

        m_dir = AppendCameraName ? Path.Combine(OutputDirectory, gameObject.name) : OutputDirectory;
        Directory.CreateDirectory(m_dir);

        if (s_active == 0) s_savedCaptureFramerate = Time.captureFramerate;
        Time.captureFramerate = Fps;
        s_active++;

        m_rt = new RenderTexture(Width, Height, 24) { antiAliasing = 4 };
        m_cam.targetTexture = m_rt;
        m_tex = new Texture2D(Width, Height, TextureFormat.RGB24, false);

        m_frame = 0;
        FramesWritten = 0;
        IsRecording = true;
        Debug.Log($"FrameSequenceRecorder [{gameObject.name}] : enregistrement -> {m_dir} @ {Fps} fps {Width}x{Height}");
    }

    public void StopRecording()
    {
        if (!IsRecording) return;
        IsRecording = false;

        s_active = Mathf.Max(0, s_active - 1);
        if (s_active == 0) Time.captureFramerate = s_savedCaptureFramerate;

        m_cam.targetTexture = null;
        if (m_rt != null) { m_rt.Release(); m_rt = null; }
        Debug.Log($"FrameSequenceRecorder [{gameObject.name}] : {FramesWritten} images écrites dans {m_dir}");
    }

    private void LateUpdate()
    {
        if (!IsRecording) return;

        var t = m_frame / (float)Fps;
        if (Duration > 0f && t > Duration)
        {
            StopRecording();
            return;
        }

        m_cam.Render();
        RenderTexture.active = m_rt;
        m_tex.ReadPixels(new Rect(0, 0, Width, Height), 0, 0);
        m_tex.Apply();
        RenderTexture.active = null;

        var bytes = Format == ImageFormat.JPG ? m_tex.EncodeToJPG(JpgQuality) : m_tex.EncodeToPNG();
        var ext = Format == ImageFormat.JPG ? "jpg" : "png";
        File.WriteAllBytes(Path.Combine(m_dir, $"frame_{m_frame:D5}.{ext}"), bytes);

        FramesWritten++;
        m_frame++;
    }

    private void OnDestroy()
    {
        if (IsRecording) StopRecording();
    }
}
