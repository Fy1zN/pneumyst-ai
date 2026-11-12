/**
 * A quick look at a chosen image before it is uploaded, to stop pictures that are clearly not chest X-rays
 * (screenshots, documents, charts, colour photos). The models score any image they are given, so without this a
 * screenshot gets "possible" findings.
 *
 * It only catches the obvious cases and runs in the browser, so it is a convenience, not a guarantee: the API still
 * accepts any readable image. Limits were set on real X-rays (with white borders, black letterboxing, burned-in labels,
 * low contrast and phone photos of film) against screenshots, charts and documents from this project.
 */

const SIZE = 256;

// Share of pixels that are strongly coloured (largest minus smallest channel above 40). X-rays stayed under 1% even as
// noisy phone photos of film; screenshots, charts and heatmaps that only this rule catches were at 4% or more.
const MAX_COLOURED = 0.02;
// Share of near-white pixels (luminance above 235). X-rays reached 37% at most; documents and app screens 67% and up.
const MAX_WHITE = 0.5;
// X-rays always have dark lungs or background: at least 4% of pixels below luminance 70 in every test image.
const MIN_DARK = 0.01;
// Dark-mode screenshots are almost entirely black; an X-ray with black side bars reached 53%.
const MAX_DARK = 0.85;

export const NOT_XRAY_TITLE = "This does not look like a chest X-ray.";

/** The reason an image is not a chest X-ray, or null when it passes. Pixels are RGBA, as from canvas getImageData. */
export function xrayProblem(pixels: Uint8ClampedArray): string | null {
  const count = pixels.length / 4;
  if (count === 0) return null;
  // Cancel the image's overall tint first, so a film photographed under warm or cool light still counts as grey.
  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    sumR += pixels[i];
    sumG += pixels[i + 1];
    sumB += pixels[i + 2];
  }
  const grey = (sumR + sumG + sumB) / 3;
  const [kr, kg, kb] = [sumR, sumG, sumB].map((sum) => (sum > 0 ? grey / sum : 1));

  let coloured = 0;
  let white = 0;
  let dark = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    const r = pixels[i];
    const g = pixels[i + 1];
    const b = pixels[i + 2];
    const [br, bg, bb] = [r * kr, g * kg, b * kb];
    if (Math.max(br, bg, bb) - Math.min(br, bg, bb) > 40) coloured++;
    const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
    if (luminance > 235) white++;
    else if (luminance < 70) dark++;
  }
  if (coloured / count > MAX_COLOURED) {
    return `${NOT_XRAY_TITLE} It has colour, and X-rays are greyscale. Use the X-ray file itself, not a screenshot or chart.`;
  }
  if (white / count > MAX_WHITE) {
    return `${NOT_XRAY_TITLE} It is mostly white, like a document or a screenshot.`;
  }
  if (dark / count < MIN_DARK) {
    return `${NOT_XRAY_TITLE} It has no dark areas where the lungs would be.`;
  }
  if (dark / count > MAX_DARK) {
    return `${NOT_XRAY_TITLE} It is almost completely dark.`;
  }
  return null;
}

/** Decodes the file at a small size and runs xrayProblem. Files the browser cannot decode pass, so the backend's own checks report them. */
export async function checkXray(file: File): Promise<string | null> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    return null;
  }
  try {
    const scale = Math.min(1, SIZE / Math.max(bitmap.width, bitmap.height));
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) return null;
    // Averaging while shrinking keeps camera noise in phone photos from counting as colour.
    context.imageSmoothingQuality = "high";
    context.drawImage(bitmap, 0, 0, width, height);
    return xrayProblem(context.getImageData(0, 0, width, height).data);
  } finally {
    bitmap.close();
  }
}