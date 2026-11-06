/**
 * Viewport-fixed placement for the usage panel.
 *
 * DSH removed `useAnchoredPosition` from its primitives after the 0.2.2-rc.1
 * release, while builds on both sides report that same version. Owning this small
 * placement keeps the badge working on either side.
 */
import { useLayoutEffect, useState, type CSSProperties, type RefObject } from 'react'

/** Anchor rectangle fields the placement reads, in viewport coordinates. */
export interface AnchorRect {
  left: number
  top: number
}

/**
 * Place a panel above its anchor, left-aligned with it, clamped into the viewport.
 * @param anchor - anchor rectangle.
 * @param panel - measured panel size.
 * @param viewport - visible viewport size.
 * @param gap - distance from the anchor's top edge.
 * @param margin - minimum distance from every viewport edge.
 * @returns fixed-position coordinates; the top and left edges win when the panel cannot fit.
 */
export function placeAbove(
  anchor: AnchorRect,
  panel: { width: number; height: number },
  viewport: { width: number; height: number },
  gap: number,
  margin: number,
): { left: number; top: number } {
  return {
    left: Math.max(margin, Math.min(anchor.left, viewport.width - margin - panel.width)),
    top: Math.max(margin, Math.min(anchor.top - gap - panel.height, viewport.height - margin - panel.height)),
  }
}

/**
 * Track a body-portaled fixed panel above its anchor while open.
 * @param options - open state, element refs, gap and viewport margin.
 * @returns coordinates, or null while closed or before the first measurement.
 */
export function useAnchoredPanel(options: {
  open: boolean
  anchorRef: RefObject<HTMLElement | null>
  panelRef: RefObject<HTMLElement | null>
  gap: number
  margin: number
}): CSSProperties | null {
  const { open, anchorRef, panelRef, gap, margin } = options
  const [position, setPosition] = useState<{ left: number; top: number } | null>(null)
  useLayoutEffect(() => {
    if (!open) {
      setPosition(null)
      return
    }
    const anchor = anchorRef.current
    const panel = panelRef.current
    const win = anchor?.ownerDocument.defaultView
    if (!anchor || !panel || !win) return
    const place = (): void => {
      const next = placeAbove(
        anchor.getBoundingClientRect(),
        { width: panel.offsetWidth, height: panel.offsetHeight },
        { width: win.innerWidth, height: win.innerHeight },
        gap,
        margin,
      )
      setPosition(current => current?.left === next.left && current.top === next.top ? current : next)
    }
    place()
    win.addEventListener('resize', place)
    win.addEventListener('scroll', place, true)
    const observer = typeof win.ResizeObserver === 'undefined' ? undefined : new win.ResizeObserver(place)
    observer?.observe(anchor)
    observer?.observe(panel)
    return () => {
      win.removeEventListener('resize', place)
      win.removeEventListener('scroll', place, true)
      observer?.disconnect()
    }
  }, [open, anchorRef, panelRef, gap, margin])
  return position
}
