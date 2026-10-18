import { Attributes } from "graphology-types";
import { NodeCircleProgram } from "sigma/rendering";

/**
 * A node program that shades each disc like a small sphere: a soft top-left
 * light, a darker silhouette toward the rim, and a faint specular glint. It is
 * a drop-in replacement for the circle fill — same attributes and vertex
 * shader (which already exposes `v_diffVector` and `v_radius`) — so it composes
 * cleanly under the border program in a compound node program.
 *
 * Output is premultiplied alpha to match Sigma's blendFunc(ONE, ONE_MINUS_SRC_ALPHA).
 */
const FRAGMENT_SHADER_SOURCE = /* glsl */ `
precision highp float;

varying vec4 v_color;
varying vec2 v_diffVector;
varying float v_radius;

uniform float u_correctionRatio;

const vec4 transparent = vec4(0.0, 0.0, 0.0, 0.0);

void main(void) {
  float aaBorder = 2.0 * u_correctionRatio;
  float dist = length(v_diffVector);

  #ifdef PICKING_MODE
  if (dist > v_radius)
    gl_FragColor = transparent;
  else
    gl_FragColor = v_color;
  #else
  // Normalised position across the disc (-1..1 on each axis).
  vec2 p = v_diffVector / v_radius;
  float r = length(p);
  float r2 = min(dot(p, p), 1.0);

  // Pseudo surface normal of a hemisphere, so lighting reads as a ball.
  float z = sqrt(1.0 - r2);
  vec3 normal = vec3(p, z);

  // Canvas Y points down, so a negative Y aims the light at the top.
  vec3 lightDir = normalize(vec3(-0.42, -0.55, 0.85));
  float diff = max(dot(normal, lightDir), 0.0);

  // Ambient + diffuse, kept in a gentle band so hues stay recognisable.
  float shade = 0.72 + 0.46 * diff;
  vec3 col = v_color.rgb * shade;

  // Darken toward the silhouette for a rounded, slightly 3D edge.
  col *= mix(1.0, 0.82, smoothstep(0.45, 1.0, r));

  // Small specular highlight near the light.
  vec3 halfDir = normalize(lightDir + vec3(0.0, 0.0, 1.0));
  float spec = pow(max(dot(normal, halfDir), 0.0), 22.0);
  col += vec3(1.0) * spec * 0.38;

  // Antialiased circular edge.
  float alpha = v_color.a * (1.0 - smoothstep(v_radius - aaBorder, v_radius, dist));

  // Premultiplied output.
  gl_FragColor = vec4(col * alpha, alpha);
  #endif
}
`;

export default class SphereNodeProgram<
  N extends Attributes = Attributes,
  E extends Attributes = Attributes,
  G extends Attributes = Attributes,
> extends NodeCircleProgram<N, E, G> {
  getDefinition() {
    return {
      ...super.getDefinition(),
      FRAGMENT_SHADER_SOURCE,
    };
  }
}
