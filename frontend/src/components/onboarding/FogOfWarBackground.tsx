import { useRef, useEffect, useCallback, useState } from "react";

/* -- Shader sources -------------------------------------------------------- */

const VERT_SRC = `
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FRAG_SRC = `
precision mediump float;

varying vec2 v_uv;
uniform vec2 u_resolution;
uniform float u_angle;
uniform sampler2D u_revealMap;

// Logo gradient: deep blue -> teal -> orange -> red
vec3 logoGradient(float t) {
  vec3 c0 = vec3(0.212, 0.380, 0.616); // #36619D
  vec3 c1 = vec3(0.184, 0.745, 0.773); // #2FBEC5
  vec3 c2 = vec3(1.000, 0.596, 0.180); // #FF982E
  vec3 c3 = vec3(0.969, 0.231, 0.255); // #F73B41
  vec3 color = mix(c0, c1, smoothstep(0.0, 0.21, t));
  color = mix(color, c2, smoothstep(0.21, 0.72, t));
  color = mix(color, c3, smoothstep(0.72, 1.0, t));
  return color;
}

void main() {
  vec2 uv = v_uv;
  float aspect = u_resolution.x / u_resolution.y;

  // Static directional gradient at random angle
  vec2 p = (uv - 0.5) * vec2(aspect, 1.0);
  vec2 dir = vec2(cos(u_angle), sin(u_angle));
  vec2 perp = vec2(-dir.y, dir.x);
  float mainT = dot(p, dir);
  float perpT = dot(p, perp);

  // Non-uniform band widths + slight perpendicular wobble
  float colorSeed = mainT * 0.8 + sin(mainT * 4.0) * 0.12 + sin(perpT * 2.5) * 0.08 + 0.5;
  vec3 gradientColor = logoGradient(clamp(colorSeed, 0.0, 1.0));

  // Brightness + saturation boost
  gradientColor *= 1.15;
  float lum = dot(gradientColor, vec3(0.299, 0.587, 0.114));
  gradientColor = mix(vec3(lum), gradientColor, 1.4);
  gradientColor = clamp(gradientColor, 0.0, 1.0);

  // Sample reveal map (flip Y: canvas 2D is Y-down, WebGL is Y-up)
  float reveal = texture2D(u_revealMap, vec2(uv.x, 1.0 - uv.y)).r;
  reveal = step(0.12, reveal); // binary: fully hidden or fully revealed

  // Mix fog and gradient
  vec3 fogColor = vec3(0.039, 0.039, 0.039); // #0a0a0a
  vec3 finalColor = mix(fogColor, gradientColor, reveal * 0.55);

  gl_FragColor = vec4(finalColor, 1.0);
}
`;

/* -- Helpers --------------------------------------------------------------- */

function compileShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

/* -- Trail canvas constants ------------------------------------------------ */

const TRAIL_SIZE = 128; // low-res for chunky pixelated reveal (~15px per cell)
const FADE_ALPHA = 0.02; // visible fade — trail disappears over ~3-4s
const BRUSH_RADIUS = 12; // in trail-canvas pixels (~140px real)
const BRUSH_CENTER_ALPHA = 0.12; // subtle per-frame reveal — needs sustained movement
const BRUSH_MID_ALPHA = 0.04;

/* -- Component ------------------------------------------------------------- */

export function FogOfWarBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouse = useRef({ x: -1, y: -1 });
  const mouseActive = useRef(false);
  const rafId = useRef<number>(0);
  const [useFallback, setUseFallback] = useState(false);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    mouse.current.x = e.clientX;
    mouse.current.y = e.clientY;
    mouseActive.current = true;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Guard: mobile or reduced motion
    const hasFinePointer = window.matchMedia("(pointer: fine)").matches;
    const prefersReduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    if (!hasFinePointer || prefersReduced) {
      setUseFallback(true);
      return;
    }

    // -- WebGL context --
    const gl = canvas.getContext("webgl", {
      alpha: false,
      antialias: false,
      preserveDrawingBuffer: false,
    });
    if (!gl) {
      setUseFallback(true);
      return;
    }

    // Handle context loss
    const handleContextLost = (e: Event) => {
      e.preventDefault();
      cancelAnimationFrame(rafId.current);
      setUseFallback(true);
    };
    canvas.addEventListener("webglcontextlost", handleContextLost);

    // -- Compile shaders --
    const vertShader = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
    const fragShader = compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
    if (!vertShader || !fragShader) {
      setUseFallback(true);
      return;
    }

    const program = gl.createProgram()!;
    gl.attachShader(program, vertShader);
    gl.attachShader(program, fragShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      setUseFallback(true);
      return;
    }
    gl.useProgram(program);

    // -- Geometry: full-screen quad --
    const positions = new Float32Array([
      -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1,
    ]);
    const posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    const aPos = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    // -- Uniforms --
    const uResolution = gl.getUniformLocation(program, "u_resolution");
    const uAngle = gl.getUniformLocation(program, "u_angle");
    const uRevealMap = gl.getUniformLocation(program, "u_revealMap");

    // Set random gradient angle once
    gl.uniform1f(uAngle!, Math.random() * Math.PI * 2);

    // -- Reveal map texture --
    const revealTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, revealTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.uniform1i(uRevealMap, 0);

    // -- Trail canvas (offscreen) --
    const trailCanvas = document.createElement("canvas");
    trailCanvas.width = TRAIL_SIZE;
    trailCanvas.height = TRAIL_SIZE;
    const trailCtx = trailCanvas.getContext("2d")!;
    trailCtx.imageSmoothingEnabled = false;
    trailCtx.fillStyle = "#000";
    trailCtx.fillRect(0, 0, TRAIL_SIZE, TRAIL_SIZE);

    // -- Resize handler --
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = window.innerWidth + "px";
      canvas.style.height = window.innerHeight + "px";
    };
    resize();
    window.addEventListener("resize", resize, { passive: true });
    window.addEventListener("mousemove", handleMouseMove, { passive: true });

    // -- Animation loop --
    const animate = () => {
      // Fade old trails
      trailCtx.fillStyle = `rgba(0, 0, 0, ${FADE_ALPHA})`;
      trailCtx.fillRect(0, 0, TRAIL_SIZE, TRAIL_SIZE);

      // Paint brush at mouse position
      if (mouseActive.current && mouse.current.x >= 0) {
        const mx = (mouse.current.x / window.innerWidth) * TRAIL_SIZE;
        const my = (mouse.current.y / window.innerHeight) * TRAIL_SIZE;
        const brushGrad = trailCtx.createRadialGradient(
          mx,
          my,
          0,
          mx,
          my,
          BRUSH_RADIUS
        );
        brushGrad.addColorStop(0, `rgba(255, 255, 255, ${BRUSH_CENTER_ALPHA})`);
        brushGrad.addColorStop(0.5, `rgba(255, 255, 255, ${BRUSH_MID_ALPHA})`);
        brushGrad.addColorStop(1, "rgba(255, 255, 255, 0)");
        trailCtx.fillStyle = brushGrad;
        trailCtx.fillRect(
          mx - BRUSH_RADIUS,
          my - BRUSH_RADIUS,
          BRUSH_RADIUS * 2,
          BRUSH_RADIUS * 2
        );
      }

      // -- Render WebGL --
      gl.viewport(0, 0, canvas.width, canvas.height);

      // Upload trail canvas as texture
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, revealTex);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        trailCanvas
      );

      gl.uniform2f(uResolution!, canvas.width, canvas.height);
      gl.drawArrays(gl.TRIANGLES, 0, 6);

      rafId.current = requestAnimationFrame(animate);
    };

    rafId.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(rafId.current);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", handleMouseMove);
      canvas.removeEventListener("webglcontextlost", handleContextLost);
      gl.deleteProgram(program);
      gl.deleteShader(vertShader);
      gl.deleteShader(fragShader);
      gl.deleteBuffer(posBuffer);
      gl.deleteTexture(revealTex);
    };
  }, [handleMouseMove]);

  if (useFallback) {
    return (
      <div
        style={{
          position: "absolute",
          inset: 0,
          pointerEvents: "none",
          background: `
            radial-gradient(ellipse at 70% 20%, rgba(234, 146, 30, 0.12) 0%, transparent 65%),
            radial-gradient(ellipse at 20% 80%, rgba(56, 189, 192, 0.08) 0%, transparent 65%),
            radial-gradient(ellipse at 50% 60%, rgba(200, 50, 80, 0.06) 0%, transparent 65%)
          `,
        }}
        aria-hidden="true"
      />
    );
  }

  return (
    <canvas
      ref={canvasRef}
      style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
      aria-hidden="true"
    />
  );
}

export default FogOfWarBackground;
