using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;
using TMPro;

public class DigitRecognizer : MonoBehaviour
{
    [Header("UI References")]
    public RawImage drawingCanvas;
    public TextMeshProUGUI resultText;
    public Button recognizeButton;
    public Button clearButton;

    // ★変更点：単一のModelAssetから配列に変更
    [Header("Sentis Model")]
    [Tooltip("使用するONNXモデルのリスト")]
    public ModelAsset[] modelAssets; 

    private Texture2D _canvasTexture;
    private Worker _engine;
    private Vector2? _lastMousePosition;
    private RectTransform _canvasRectTransform;
    private int activeModelIndex = 0;

    void Start()
    {
        _canvasTexture = new Texture2D(256, 256, TextureFormat.RGBA32, false);
        drawingCanvas.texture = _canvasTexture;
        _canvasRectTransform = drawingCanvas.rectTransform;

        ClearCanvas();

        // ★変更点：最初に0番目のモデルをロードする
        if (modelAssets != null && modelAssets.Length > 0)
        {
            var model = ModelLoader.Load(modelAssets[0]);
            _engine = new Worker(model, BackendType.GPUCompute);
            activeModelIndex = 0;
            Debug.Log($"初期モデルとして {modelAssets[0].name} をロードしました。");
        }
        else
        {
            Debug.LogError("モデルがインスペクターに設定されていません！");
        }

        recognizeButton.onClick.AddListener(RecognizeDigit);
        clearButton.onClick.AddListener(ClearCanvas);
    }

    // ★追加点：モデルを切り替えるための公開メソッド
    public void ChangeModel(int modelIndex)
    {
        // インデックスが有効な範囲にあるか確認
        if (modelIndex < 0 || modelIndex >= modelAssets.Length || modelIndex == activeModelIndex)
        {
            return;
        }

        Debug.Log($"モデルを {modelAssets[activeModelIndex].name} から {modelAssets[modelIndex].name} に変更します。");

        // 1. 既存の推論エンジンを破棄する (非常に重要)
        _engine?.Dispose();

        // 2. 新しいモデルをロードする
        var newModel = ModelLoader.Load(modelAssets[modelIndex]);

        // 3. 新しいモデルで推論エンジンを再作成する
        _engine = new Worker(newModel, BackendType.GPUCompute);
        
        // 4. 現在のモデルインデックスを更新
        activeModelIndex = modelIndex;
        
        Debug.Log("モデルの変更が完了しました。");
    }

    void Update()
    {
        if (Input.GetMouseButton(0))
        {
            HandleDrawing();
        }
        else
        {
            _lastMousePosition = null;
        }

        if (Input.GetKeyDown(KeyCode.P))
        {
            RecognizeDigit();
        }
    }

    private void HandleDrawing()
    {
        if (RectTransformUtility.ScreenPointToLocalPointInRectangle(
            _canvasRectTransform, Input.mousePosition, null, out Vector2 localPoint))
        {
            Vector2 pivot = _canvasRectTransform.pivot;
            localPoint.x += pivot.x * _canvasRectTransform.rect.width;
            localPoint.y += pivot.y * _canvasRectTransform.rect.height;

            Vector2 textureCoord = new Vector2(
                (localPoint.x / _canvasRectTransform.rect.width) * _canvasTexture.width,
                (localPoint.y / _canvasRectTransform.rect.height) * _canvasTexture.height
            );

            if (_lastMousePosition.HasValue)
            {
                DrawLine(_lastMousePosition.Value, textureCoord, 10, Color.white);
            }
            _lastMousePosition = textureCoord;
            _canvasTexture.Apply();
        }
    }

    private void DrawLine(Vector2 from, Vector2 to, int thickness, Color color)
    {
        int x0 = (int)from.x;
        int y0 = (int)from.y;
        int x1 = (int)to.x;
        int y1 = (int)to.y;
        int dx = Mathf.Abs(x1 - x0);
        int dy = Mathf.Abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int err = dx - dy;

        while (true)
        {
            for (int i = -thickness; i <= thickness; i++)
            {
                for (int j = -thickness; j <= thickness; j++)
                {
                    if (i * i + j * j <= thickness * thickness)
                    {
                        if(x0 + i >= 0 && x0 + i < _canvasTexture.width && y0 + j >= 0 && y0 + j < _canvasTexture.height)
                        {
                            _canvasTexture.SetPixel(x0 + i, y0 + j, color);
                        }
                    }
                }
            }
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
    }

    public void RecognizeDigit()
    {
        var transform = new TextureTransform()
            .SetDimensions(28, 28, 1)
            .SetTensorLayout(TensorLayout.NHWC);

        using var inputTensor = TextureConverter.ToTensor(_canvasTexture, transform);
        _engine.Schedule(inputTensor);

        var outputGpu = _engine.PeekOutput() as Tensor<float>;
        if (outputGpu == null)
        {
            Debug.LogError("出力テンソルが取得できませんでした。");
            return;
        }
        using var outputTensor = outputGpu.ReadbackAndClone();

        float maxProbability = -1f;
        int predictedDigit = -1;
        int outputLength = outputTensor.shape.length;
        for (int i = 0; i < outputLength; i++)
        {
            float prob = outputTensor[i];
            if (prob > maxProbability)
            {
                maxProbability = prob;
                predictedDigit = i;
            }
        }
        resultText.text = $"predict: {predictedDigit}";
        Debug.Log($"Predicted: {predictedDigit} with probability: {maxProbability}");
    }

    public void ClearCanvas()
    {
        Color32[] pixels = new Color32[_canvasTexture.width * _canvasTexture.height];
        System.Array.Fill(pixels, Color.black);
        _canvasTexture.SetPixels32(pixels);
        _canvasTexture.Apply();
        resultText.text = "spell numbers here";
        _lastMousePosition = null;
    }

    void OnDestroy()
    {
        _engine?.Dispose();
    }
}