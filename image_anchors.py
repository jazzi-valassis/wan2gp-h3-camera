"""Preserve native image order while adding or replacing timed conditions."""


def merge_timed_images(images, positions, additions):
    """Return independent galleries/positions and stable identities for relabeling.

    Positions are zero based, already resolved by the host. Native injection
    consumes the first N gallery images, leaving the rest as references.
    Nonnegative identities track the old gallery; negative ones identify additions.
    """
    images, positions = list(images), list(positions)
    if len(positions) > len(images):
        raise ValueError("Each injected frame needs an image.")
    identities = list(range(len(images)))
    for frame, image in sorted(additions.items()):
        matches = [index for index, existing in enumerate(positions) if existing == frame]
        if matches:
            # The native host uses the last supplied image for duplicate times.
            images[matches[-1]] = image
        else:
            index = len(positions)
            images.insert(index, image)
            identities.insert(index, -frame - 1)
            positions.append(frame)
    return images, positions, identities
