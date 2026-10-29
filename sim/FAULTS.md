# Faults and violations

A scenario lists fault keys with a per-item probability (`faults:` in the YAML). Each item draws its faults from the `faults` stream of its seed, so the same seed always gives the same faults. A station whose fault fires skips its part; the item then carries the violation keys below in `ground_truth.json` (`expected_violations`).

Violation keys follow the semantics of the rule engine: a checkpoint looks at the item within 1.0 s of the moment it is judged, and "leaves the view" is the item's last sighting. An item last seen within 1.0 s of the end of the footage has not left the view (`EXIT_MARGIN_S`).

## S1 jar line (`jar_line`)

| Fault | Station | Rendered as | Violation key | Rule sentence |
|---|---|---|---|---|
| `missing_cap` | capper | the jar passes the capper with no cap | `cap_and_label_on_exit` | Every jar needs a cap and a label before it leaves the view. |
| `missing_label` | labeller | the jar passes the labeller with no label sleeve | `cap_and_label_on_exit` | Every jar needs a cap and a label before it leaves the view. |
| jam (`jam_probability`, `jams`) | whole belt | the belt and every jar stop for 7–14 s | `capper_dwell` on each jar that stays in the capper zone more than 5 s | A jar must not stay in the capper zone for more than 5 seconds. |

"At least 20 jars per minute must pass" is a line-level rule, not an item violation: scoring derives the rate from each item's `t_cross.view_exit`. A jam drops it below 20.

## S2 carton packing (`carton_packing`)

| Fault | Station | Rendered as | Violation key | Rule sentence |
|---|---|---|---|---|
| `flap_up` | folder | one side flap (`open_flap`: front or back, the ones facing the camera) stays up, the other three fold | `flaps_before_tape` (if the carton leaves the view) | The carton must be closed with all 4 flaps down before tape. |
| `missing_label` | labeller | no shipping label on the front face | `label_at_sealing`, `shipping_label_qr_at_end` | Every carton needs a label when it enters sealing. / Every carton needs a shipping label and a QR code at the end of the footage. |
| `missing_qr` | labeller | the label is printed without its QR code | `shipping_label_qr_at_end` | Every carton needs a shipping label and a QR code at the end of the footage. |
| `missing_tape` | taper | no tape strip along the top | `tape_on_exit` (if the carton leaves the view) | Every carton must have tape before it leaves the view. |

`shipping_label_qr_at_end` judges every carton as last seen, so cartons still upstream of the labeller when the footage ends carry it too. This follows the engine's current semantics; whether that is the intended reading is an open question for the owner.

Jams in S2 stop the belt but no S2 rule depends on them; they appear only as `jam` events.
